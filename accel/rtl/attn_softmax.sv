// Attention softmax unit (H5, stage 1: scores come from and probabilities go to DDR3).
//
// Reads the raw INT32 scores S written by the engine in RAW32 mode (row-major, `s_stride` bytes between rows),
// applies the model's integer softmax to each row (bit-exact with export_check.softmax, see ops.c
// softmax_row_int) and writes P (UQ0.16) in the activation format of the engine, ready to be the A operand of
// P.V:   [subtile of 16 rows][plane][k][16 bytes],  plane 0 = high byte, plane 1 = low byte,  k = 0 .. kp-1.
//
// Per row: pass 1 (maximum) while the row streams in, pass 2 (exponent index -> EXP table -> sum), reciprocal
// of the sum, pass 3 (normalise -> 16 rows of P kept in 16 small RAMs). After 16 rows (or the last row) the
// RAMs are read out transposed, 16 rows per 16-byte word, and written with single-beat writes. Columns >= cols
// (padding keys) are masked: they add nothing to the maximum or to the sum and their P is zero, as are the
// k positions from cols to kp-1 and the rows past the last row.
//
// Control is by plain inputs (the CSR block of fpgate_accel_core drives them); `done` and `err_valid` are
// one-cycle pulses. Error codes: 2 bad parameters, 3 read timeout, 4 write timeout, 6 watchdog.
module attn_softmax
#(
  parameter int LANES2      = 4,        // columns per cycle in the exponent pass (4 or 8)
  parameter int ROW_WORDS   = 130,      // 256-bit words per score row buffer (>= ceil(cols / 8))
  parameter int BURST       = 16,
  parameter int DMA_TIMEOUT = 4096,
  parameter int WD_LIMIT    = 1 << 20
)(
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  input  logic         soft_reset,
  input  logic [15:0]  rows,            // query rows to process
  input  logic [15:0]  cols,            // valid keys per row (1025)
  input  logic [15:0]  kp,              // k extent of P (even, cols <= kp <= 8 * ROW_WORDS), e.g. 1026
  input  logic [31:0]  s_addr,          // byte address of the score matrix (32-byte aligned)
  input  logic [31:0]  s_stride,        // bytes between score rows (multiple of 32)
  input  logic [31:0]  p_addr,          // byte address of the P image (32-byte aligned)
  input  logic [30:0]  sm_m0,
  input  logic [5:0]   sm_sh,
  output logic         busy,
  output logic         done,
  output logic         err_valid,
  output logic [7:0]   err_code,
  output logic [31:0]  bytes_rd,
  output logic [31:0]  bytes_wr,
  // Avalon-MM read master
  output logic [31:0]  rd_address,
  output logic         rd_read,
  output logic [7:0]   rd_burstcount,
  input  logic         rd_waitrequest,
  input  logic [255:0] rd_readdata,
  input  logic         rd_readdatavalid,
  // Avalon-MM write master
  output logic [31:0]  wr_address,
  output logic         wr_write,
  output logic [255:0] wr_writedata,
  output logic [31:0]  wr_byteenable,
  output logic [7:0]   wr_burstcount,
  input  logic         wr_waitrequest
);
  localparam int AW      = $clog2(ROW_WORDS);
  localparam int STEPS   = 8 / LANES2;       // cycles per 8-column word in pass 2
  localparam int IDX_LAT = 9;                // softmax_idx_lane latency
  localparam int TAG_LAT = IDX_LAT + 1;      // + exp ROM read
  localparam int P3_LAT  = 5;                // softmax_p3_lane latency

  wire rst_dp = rst_n & ~soft_reset;

  typedef enum logic [4:0] {
    S_IDLE, S_CHK, S_REQ, S_RD, S_MAXD, S_P2, S_P2D, S_RCP, S_P3, S_P3D, S_NEXT, S_OUT, S_OUTD, S_FIN, S_ERR
  } state_t;
  state_t state, state_q;

  // ---- operation registers ------------------------------------------------------------------
  logic [15:0] rows_q, cols_q, kp_q;
  logic [31:0] s_cur, s_stride_q, p_grp, p_addr_q;
  logic [30:0] m0_q;
  logic [32:0] m0x3_q;
  logic [5:0]  sh_q;
  logic [53:0] rnd_q;
  logic [15:0] nw;                          // 8-column words per row = ceil(cols / 8)
  logic [15:0] row_cnt;                     // rows finished
  logic [4:0]  grp_row;                     // row inside the current group of 16
  logic [4:0]  grp_n;                       // valid rows of the group being written out
  logic [7:0]  next_err;
  logic [31:0] wd;

  // ---- DMA ----------------------------------------------------------------------------------------
  logic         rd_cmd_ready, rd_busy, rd_done, rd_err;
  logic         beat_valid;
  logic [255:0] beat_data;
  logic [15:0]  beat_idx;
  wire          rd_cmd_valid = (state == S_REQ);

  // the DMAs survive a soft reset / error: they finish the bus transaction in flight (see dma_wr, dma_rd)
  wire dma_flush = soft_reset | (state == S_ERR);

  dma_rd #(.BURST(BURST), .TIMEOUT(DMA_TIMEOUT)) u_dma_rd (
    .clk, .rst_n, .flush(dma_flush),
    .cmd_valid(rd_cmd_valid), .cmd_ready(rd_cmd_ready), .cmd_addr(s_cur), .cmd_beats(nw),
    .busy(rd_busy), .done(rd_done), .err(rd_err),
    .beat_valid, .beat_data, .beat_idx,
    .m_address(rd_address), .m_read(rd_read), .m_burstcount(rd_burstcount),
    .m_waitrequest(rd_waitrequest), .m_readdata(rd_readdata), .m_readdatavalid(rd_readdatavalid)
  );

  logic         wr_in_valid, wr_in_ready, wr_busy, wr_beat, wr_err;
  logic [31:0]  wr_in_addr;
  logic [255:0] wr_in_data;

  dma_wr #(.TIMEOUT(DMA_TIMEOUT)) u_dma_wr (
    .clk, .rst_n, .flush(dma_flush),
    .in_valid(wr_in_valid), .in_ready(wr_in_ready),
    .in_addr(wr_in_addr), .in_data(wr_in_data), .in_be(32'hFFFF_FFFF),
    .busy(wr_busy), .wr_beat, .err(wr_err),
    .m_address(wr_address), .m_write(wr_write), .m_writedata(wr_writedata),
    .m_byteenable(wr_byteenable), .m_burstcount(wr_burstcount), .m_waitrequest(wr_waitrequest)
  );

  // ---- score row buffer ---------------------------------------------------------------------------------
  logic [255:0] srow [ROW_WORDS];
  logic [255:0] srow_q;
  logic [AW-1:0] p2_raddr;
  // the data from the HPS port FIFO is registered once before it fans out to the RAM and the maximum tree
  logic         bv_q, rdone_q;
  logic [255:0] bd_q;
  logic [15:0]  bi_q;
  always_ff @(posedge clk) begin
    bv_q <= beat_valid; bd_q <= beat_data; bi_q <= beat_idx; rdone_q <= rd_done;
  end

  always_ff @(posedge clk) begin
    if (bv_q && state == S_RD) srow[bi_q[AW-1:0]] <= bd_q;
    srow_q <= srow[p2_raddr];
  end

  // ---- pass 1: maximum over the valid columns of the streaming row -------------------------------------------
  logic signed [31:0] mx_run;
  logic               m_v1, m_v2, m_v3;
  logic signed [31:0] m_a [4];
  logic signed [31:0] m_b [2];
  logic signed [31:0] m_c;

  function automatic logic signed [31:0] smax(input logic signed [31:0] a, input logic signed [31:0] b);
    return (a > b) ? a : b;
  endfunction

  always_ff @(posedge clk) begin
    logic signed [31:0] w8 [8];
    for (int l = 0; l < 8; l++)
      w8[l] = ((int'(bi_q) * 8 + l) < int'(cols_q)) ? bd_q[32*l +: 32] : 32'sh8000_0000;
    m_v1 <= bv_q && (state == S_RD);
    for (int i = 0; i < 4; i++) m_a[i] <= smax(w8[2*i], w8[2*i+1]);
    m_v2 <= m_v1;
    for (int i = 0; i < 2; i++) m_b[i] <= smax(m_a[2*i], m_a[2*i+1]);
    m_v3 <= m_v2;
    m_c <= smax(m_b[0], m_b[1]);
    if (state == S_REQ) mx_run <= 32'sh8000_0000;
    else if (m_v3)      mx_run <= smax(mx_run, m_c);
  end

  // ---- pass 2: exponent index lanes, EXP ROM, sum, e buffer ---------------------------------------------------
  logic [15:0]   p2_i;                       // half-step counter
  logic          p2_issue, rd_v1;
  logic [7:0]    word_q;
  logic          half_q;
  logic          lane_in_v;
  logic          ls_v, ls_h;
  logic [7:0]    ls_w;
  logic [LANES2-1:0] ls_cv;
  logic signed [31:0] ls_s [LANES2];
  logic          lane_v [LANES2];
  logic [10:0]   lane_idx [LANES2];
  logic [15:0]   rom_q [LANES2];

  // tags follow the lanes + ROM
  logic          tag_v   [TAG_LAT];
  logic [7:0]    tag_w   [TAG_LAT];
  logic          tag_h   [TAG_LAT];
  logic [LANES2-1:0] tag_cv [TAG_LAT];
  logic [LANES2-1:0] cv_now;

  always_comb begin
    p2_raddr = AW'(p2_i / STEPS);
    for (int l = 0; l < LANES2; l++)
      cv_now[l] = ((int'(word_q) * 8 + int'(half_q) * LANES2 + l) < int'(cols_q));
  end

  always_ff @(posedge clk) begin
    rd_v1  <= (state == S_P2);
    word_q <= 8'(p2_i / STEPS);
    half_q <= (STEPS == 2) ? p2_i[0] : 1'b0;
    // lane selection registered: the 256-bit RAM output is narrowed to the lanes here
    ls_v  <= rd_v1;
    ls_w  <= word_q;
    ls_h  <= half_q;
    ls_cv <= cv_now;
    for (int l = 0; l < LANES2; l++) ls_s[l] <= srow_q[32*(int'(half_q) * LANES2 + l) +: 32];
    tag_v[0]  <= ls_v;
    tag_w[0]  <= ls_w;
    tag_h[0]  <= ls_h;
    tag_cv[0] <= ls_cv;
    for (int d = 1; d < TAG_LAT; d++) begin
      tag_v[d]  <= tag_v[d-1];
      tag_w[d]  <= tag_w[d-1];
      tag_h[d]  <= tag_h[d-1];
      tag_cv[d] <= tag_cv[d-1];
    end
  end
  assign lane_in_v = ls_v;

  genvar gl;
  generate
    for (gl = 0; gl < LANES2; gl++) begin : g_idx
      softmax_idx_lane u_lane (
        .clk, .rst_n(rst_dp),
        .in_valid(lane_in_v),
        .s(ls_s[gl]),
        .mx(mx_run),
        .m0(m0_q), .m0x3(m0x3_q), .sh(sh_q), .rnd(rnd_q),
        .out_valid(lane_v[gl]), .idx(lane_idx[gl])
      );
    end
    for (gl = 0; gl < LANES2 / 2; gl++) begin : g_exp
      softmax_exp_rom u_rom (
        .clk,
        .addr0(lane_idx[2*gl]),   .q0(rom_q[2*gl]),
        .addr1(lane_idx[2*gl+1]), .q1(rom_q[2*gl+1])
      );
    end
  endgenerate

  // exponentials: masked (one register stage), summed, stored
  logic [15:0]  e_q [LANES2];
  logic         eq_v, eq_h;
  logic [7:0]   eq_w;
  logic [26:0]  sum_acc;
  logic [26:0]  sum_step;
  logic         sum_v;
  logic [127:0] ebuf [ROW_WORDS];
  logic [127:0] ebuf_q;
  logic [AW-1:0] p3_raddr;

  always_ff @(posedge clk) begin
    for (int l = 0; l < LANES2; l++) e_q[l] <= tag_cv[TAG_LAT-1][l] ? rom_q[l] : 16'd0;
    eq_v <= tag_v[TAG_LAT-1];
    eq_w <= tag_w[TAG_LAT-1];
    eq_h <= tag_h[TAG_LAT-1];
  end

  always_ff @(posedge clk) begin
    logic [26:0] s_tmp;
    s_tmp = '0;
    for (int l = 0; l < LANES2; l++) s_tmp = s_tmp + 27'(e_q[l]);
    sum_v    <= eq_v;
    sum_step <= s_tmp;
    if (state == S_P2 && p2_i == 0) sum_acc <= '0;      // new row (cleared before the first result arrives)
    else if (sum_v)                  sum_acc <= sum_acc + sum_step;
    ebuf_q <= ebuf[p3_raddr];
  end

  generate
    if (LANES2 == 4) begin : g_ew
      logic [63:0] e_lo_hold;
      always_ff @(posedge clk) if (eq_v) begin
        if (!eq_h) e_lo_hold <= {e_q[3], e_q[2], e_q[1], e_q[0]};
        else ebuf[AW'(eq_w)] <= {e_q[3], e_q[2], e_q[1], e_q[0], e_lo_hold};
      end
    end else begin : g_ew
      always_ff @(posedge clk) if (eq_v)
        ebuf[AW'(eq_w)] <= {e_q[LANES2-1], e_q[LANES2-2], e_q[LANES2-3], e_q[LANES2-4],
                            e_q[LANES2-5], e_q[LANES2-6], e_q[LANES2-7], e_q[LANES2-8]};
    end
  endgenerate

  // ---- reciprocal of the sum ----------------------------------------------------------------------------
  logic [4:0]  kb;
  logic [7:0]  mant;
  logic [15:0] rcp_q;
  logic [14:0] r_q;
  logic [4:0]  sh3_q;
  logic [25:0] rnd3_q;
  logic [2:0]  rcp_cnt;

  function automatic logic [4:0] lead_one(input logic [26:0] v);
    logic [4:0] r;
    r = '0;
    for (int i = 0; i < 27; i++) if (v[i]) r = 5'(i);
    return r;
  endfunction

  softmax_recip_rom u_rcp (.clk, .addr0(mant), .q0(rcp_q));

  always_ff @(posedge clk) begin
    kb   <= lead_one(sum_acc);
    mant <= 8'((sum_acc >> (kb - 5'd8)));       // bits [7:0] of the 9-bit mantissa (the top one is dropped)
    r_q    <= rcp_q[14:0];
    sh3_q  <= kb - 5'd1;
    rnd3_q <= 26'd1 << (kb - 5'd2);
  end

  // ---- pass 3: normalise, into the row RAMs -------------------------------------------------------------------
  logic [AW:0]  p3_i;
  logic         p3_issue, p3_v1;
  logic [7:0]   p3_w1;
  logic         p3_lv [8];
  logic [15:0]  p3_p  [8];
  logic         p3_tv [P3_LAT];
  logic [7:0]   p3_tw [P3_LAT];

  assign p3_raddr = AW'(p3_i);

  always_ff @(posedge clk) begin
    p3_v1 <= (state == S_P3);
    p3_w1 <= 8'(p3_i);
    p3_tv[0] <= p3_v1;
    p3_tw[0] <= p3_w1;
    for (int d = 1; d < P3_LAT; d++) begin
      p3_tv[d] <= p3_tv[d-1];
      p3_tw[d] <= p3_tw[d-1];
    end
  end

  generate
    for (gl = 0; gl < 8; gl++) begin : g_p3
      softmax_p3_lane u_lane (
        .clk, .rst_n(rst_dp),
        .in_valid(p3_v1), .e(ebuf_q[16*gl +: 16]), .r(r_q), .sh(sh3_q), .rnd(rnd3_q),
        .out_valid(p3_lv[gl]), .p(p3_p[gl])
      );
    end
  endgenerate

  logic [127:0] p3_word;
  always_comb begin
    for (int l = 0; l < 8; l++) p3_word[16*l +: 16] = p3_p[l];
  end

  // 16 row RAMs: one 128-bit word (8 columns) per address
  logic [127:0] pram_q [16];
  logic [AW-1:0] out_a;
  generate
    for (gl = 0; gl < 16; gl++) begin : g_pram
      softmax_rowram #(.DEPTH(ROW_WORDS), .AW(AW)) u_ram (
        .clk,
        .we(p3_tv[P3_LAT-1] && grp_row == 5'(gl)), .waddr(AW'(p3_tw[P3_LAT-1])), .wdata(p3_word),
        .raddr(out_a), .q(pram_q[gl])
      );
    end
  endgenerate

  // ---- output: transposed read-out of the 16 row RAMs ------------------------------------------------------------
  logic [AW:0]  o_a;                        // 8-column word being read out
  logic [3:0]   o_b;                        // beat inside the word: plane = b[2], pair = b[1:0]
  logic [1:0]   o_wait;
  logic         o_active;
  logic [15:0]  o_pair;                     // k pair index of the current beat
  logic [255:0] o_data;
  logic         o_skip;

  always_comb begin
    out_a  = AW'(o_a);
    o_pair = 16'({o_a, 2'b00}) + 16'(o_b[1:0]);
    o_skip = (o_pair >= (kp_q >> 1));
    for (int i = 0; i < 16; i++) begin
      logic [15:0] v0, v1;
      v0 = pram_q[i][16*(2*int'(o_b[1:0])) +: 16];
      v1 = pram_q[i][16*(2*int'(o_b[1:0]) + 1) +: 16];
      o_data[8*i +: 8]       = (i < int'(grp_n)) ? (o_b[2] ? v0[7:0] : v0[15:8]) : 8'd0;
      o_data[128 + 8*i +: 8] = (i < int'(grp_n)) ? (o_b[2] ? v1[7:0] : v1[15:8]) : 8'd0;
    end
  end

  // registered output stage: the beat (address, data) is built here and offered to the write DMA from registers
  logic         ov;
  logic [31:0]  oaddr;
  logic [255:0] odata;
  wire          o_addr_plane = o_b[2];
  wire [31:0]   o_addr_now = p_grp + (o_addr_plane ? 32'(kp_q) * 32'd16 : 32'd0) + 32'(o_pair) * 32'd32;
  wire          can_load = !ov || wr_in_ready;
  wire          o_do = (state == S_OUT) && o_active && (o_wait == 2'd0);
  wire          o_load = o_do && !o_skip && can_load;

  always_ff @(posedge clk) begin
    if (!rst_dp) ov <= 1'b0;
    else begin
      if (ov && wr_in_ready) ov <= 1'b0;
      if (o_load) begin
        ov    <= 1'b1;
        oaddr <= o_addr_now;
        odata <= o_data;
      end
    end
  end
  assign wr_in_valid = ov;
  assign wr_in_addr  = oaddr;
  assign wr_in_data  = odata;

  // ---- control --------------------------------------------------------------------------------------------------------
  logic [7:0] drain;
  logic       err_pulse;

  always_ff @(posedge clk) begin
    done      <= 1'b0;
    err_valid <= 1'b0;
    state_q   <= state;
    p2_issue  <= 1'b0;
    p3_issue  <= 1'b0;
    if (!rst_dp) begin
      state    <= S_IDLE;
      wd       <= '0;
      o_active <= 1'b0;
    end else begin
      if (state == S_IDLE || beat_valid || wr_beat || p2_issue || p3_issue || state != state_q) wd <= '0;
      else wd <= wd + 1'b1;
      if (wd >= 32'(WD_LIMIT) && state != S_IDLE) begin next_err <= 8'd6; state <= S_ERR; end
      else if (rd_err && state != S_IDLE)       begin next_err <= 8'd3; state <= S_ERR; end
      else if (wr_err && state != S_IDLE)       begin next_err <= 8'd4; state <= S_ERR; end
      else case (state)
        S_IDLE: if (start) begin
          rows_q <= rows; cols_q <= cols; kp_q <= kp;
          s_cur <= s_addr; s_stride_q <= s_stride; p_grp <= p_addr; p_addr_q <= p_addr;
          m0_q <= sm_m0; m0x3_q <= {2'b0, sm_m0} + {1'b0, sm_m0, 1'b0}; sh_q <= sm_sh;
          rnd_q <= 54'd1 << (sm_sh - 6'd1);
          nw <= (cols + 16'd7) >> 3;
          row_cnt <= '0; grp_row <= '0;
          state <= S_CHK;
        end
        S_CHK: begin
          if (rows_q == 0 || cols_q == 0 || cols_q > 16'(8 * ROW_WORDS) || kp_q[0] || kp_q < cols_q ||
              kp_q > 16'(8 * ROW_WORDS) || sh_q == 0 || sh_q > 6'd46 ||
              s_cur[4:0] != 0 || p_grp[4:0] != 0 || s_stride_q[4:0] != 0) begin
            next_err <= 8'd2; state <= S_ERR;
          end else state <= S_REQ;
        end
        S_REQ: if (rd_cmd_ready) state <= S_RD;
        S_RD: if (rdone_q) begin drain <= 8'd6; state <= S_MAXD; end
        S_MAXD: begin
          if (drain == 0) begin p2_i <= '0; state <= S_P2; end
          else drain <= drain - 1'b1;
        end
        S_P2: begin
          // sum_acc is cleared when p2_i == 0 (here), the first issue follows
          p2_issue <= 1'b1;
          if (p2_i == nw * STEPS - 1) begin drain <= 8'(TAG_LAT + 7); state <= S_P2D; end
          p2_i <= p2_i + 1'b1;
        end
        S_P2D: begin
          p2_issue <= 1'b0;
          if (drain == 0) begin rcp_cnt <= '0; state <= S_RCP; end
          else drain <= drain - 1'b1;
        end
        S_RCP: begin
          rcp_cnt <= rcp_cnt + 1'b1;
          if (rcp_cnt == 3'd5) begin p3_i <= '0; state <= S_P3; end
        end
        S_P3: begin
          p3_issue <= 1'b1;
          p3_i <= p3_i + 1'b1;
          if (p3_i == (AW+1)'(nw - 1)) begin drain <= 8'(P3_LAT + 7); state <= S_P3D; end
        end
        S_P3D: begin
          p3_issue <= 1'b0;
          if (drain == 0) state <= S_NEXT;
          else drain <= drain - 1'b1;
        end
        S_NEXT: begin
          row_cnt <= row_cnt + 1'b1;
          s_cur   <= s_cur + s_stride_q;
          if (grp_row == 5'd15 || row_cnt + 16'd1 == rows_q) begin
            grp_n <= grp_row + 5'd1;
            o_a <= '0; o_b <= '0; o_wait <= 2'd2; o_active <= 1'b1;
            state <= S_OUT;
          end else begin
            grp_row <= grp_row + 1'b1;
            state <= S_REQ;
          end
        end
        S_OUT: begin
          if (o_wait != 0) o_wait <= o_wait - 1'b1;
          else if (o_active) begin
            if (o_skip || can_load) begin
              if (o_b == 4'd7) begin
                o_b <= '0;
                o_wait <= 2'd2;
                if (o_a == (AW+1)'(nw - 1)) begin o_active <= 1'b0; state <= S_OUTD; end
                else o_a <= o_a + 1'b1;
              end else o_b <= o_b + 1'b1;
            end
          end
        end
        S_OUTD: if (!wr_busy && !ov) begin
          grp_row <= '0;
          p_grp   <= p_grp + 32'(kp_q) * 32'd32;       // 2 planes x kp words x 16 bytes
          state   <= (row_cnt == rows_q) ? S_FIN : S_REQ;
        end
        S_FIN: begin done <= 1'b1; state <= S_IDLE; end
        S_ERR: begin err_valid <= 1'b1; err_code <= next_err; state <= S_IDLE; end
        default: state <= S_IDLE;
      endcase
    end
  end

  assign busy = (state != S_IDLE);

  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      bytes_rd <= '0;
      bytes_wr <= '0;
    end else begin
      if (beat_valid) bytes_rd <= bytes_rd + 32'd32;
      if (wr_beat)    bytes_wr <= bytes_wr + 32'd32;
    end
  end
endmodule
