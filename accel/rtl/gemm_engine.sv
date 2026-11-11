// GEMM engine: tile sequencer + read/write DMA + on-chip buffers + operand feeder +
// linear_tile (array + epilogue) + output row writer. One operation (M x N x K) is
// started with `start` and the CSR-style parameters; the result is written to memory
// requantized, row by row, with a stride between rows (see DECISIONES.md, "H2").
//
// Memory interfaces: two Avalon-MM masters (read and write), 256-bit data, byte
// addresses; Platform Designer arbitrates them onto the single HPS SDRAM port.
//
// OP: [0] 16-bit input, [1] signed input, [2] 16-bit output, [3] raw INT32 output (no epilogue: each
//     row of a 16-column tile is 64 bytes, two beats, at ADDR_OUT + row * STRIDE_OUT + tile * 64),
//     [15:8] subtiles per panel.
module gemm_engine
  import fpgate_pkg::*;
#(
  parameter int A_BEATS     = 8192,
  parameter int W_BEATS     = 1024,
  parameter int LANES       = 4,
  parameter int OUT_FIFO    = 32,
  parameter int WD_LIMIT    = 1 << 20,
  parameter int DMA_TIMEOUT = 4096,
  parameter int BURST       = 16
)(
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  input  logic         soft_reset,
  input  logic [31:0]  op,
  input  logic [31:0]  dim_m, dim_n, dim_k,
  input  logic [31:0]  addr_a, addr_w, addr_out, stride_out,
  output logic         hw_done,
  output logic         hw_err_valid,
  output logic [7:0]   hw_err_code,
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
  localparam int ROWS = PE_ROWS;
  localparam int COLS = PE_COLS;

  wire in16    = op[0];
  wire insign  = op[1];
  wire out16   = op[2];
  wire raw32   = op[3];
  wire [7:0] spp = op[15:8];

  // ---- resets ---------------------------------------------------------------
  logic abort;
  wire  rst_seq = rst_n & ~soft_reset;
  wire  rst_dp  = rst_n & ~soft_reset & ~abort;

  // ---- sequencer --------------------------------------------------------------
  logic        rd_cmd_valid, rd_cmd_ready, rd_to_a, rd_done, rd_err;
  logic [31:0] rd_cmd_addr;
  logic [15:0] rd_cmd_beats;
  logic        w_load_sel, w_comp_sel;
  logic        fd_start, fd_two_pass, sub_start;
  logic [15:0] fd_a_base, fd_k;
  logic [31:0] sub_addr;
  logic [4:0]  sub_rows_valid;
  logic        row_last_seen, lt_busy, wr_idle, progress, ext_err_valid;
  logic [7:0]  ext_err_code;
  logic        seq_busy;

  tile_sequencer #(.A_WORDS(A_BEATS * 2), .W_BEATS(W_BEATS), .WD_LIMIT(WD_LIMIT)) u_seq (
    .clk, .rst_n(rst_seq), .start,
    .op_in16(in16), .op_out16(out16), .op_raw32(raw32), .sub_per_panel(spp),
    .dim_m, .dim_n, .dim_k, .addr_a, .addr_w, .addr_out, .stride_out,
    .rd_cmd_valid, .rd_cmd_ready, .rd_cmd_addr, .rd_cmd_beats, .rd_to_a, .rd_done, .rd_err,
    .w_load_sel, .w_comp_sel,
    .fd_start, .fd_a_base, .fd_k, .fd_two_pass,
    .sub_start, .sub_addr, .sub_rows_valid,
    .row_last_seen, .lt_busy, .wr_idle, .progress, .ext_err_valid, .ext_err_code,
    .busy(seq_busy), .done(hw_done), .err_valid(hw_err_valid), .err_code(hw_err_code), .abort
  );

  // ---- read DMA and buffer fill --------------------------------------------------
  logic         beat_valid;
  logic [255:0] beat_data;
  logic [15:0]  beat_idx;
  logic         dma_rd_busy, tgt_a, rd_done_dma;

  // The buffers register their write side WR_PIPE times; the sequencer must not see "done"
  // before the last beat is really in the RAM (one extra cycle of margin).
  localparam int RD_DONE_DLY = 3;
  logic [RD_DONE_DLY-1:0] rd_done_sr;
  always_ff @(posedge clk) begin
    if (!rst_dp) rd_done_sr <= '0;
    else         rd_done_sr <= {rd_done_sr[RD_DONE_DLY-2:0], rd_done_dma};
  end
  assign rd_done = rd_done_sr[RD_DONE_DLY-1];

  dma_rd #(.BURST(BURST), .TIMEOUT(DMA_TIMEOUT)) u_dma_rd (
    .clk, .rst_n(rst_dp),
    .cmd_valid(rd_cmd_valid), .cmd_ready(rd_cmd_ready),
    .cmd_addr(rd_cmd_addr), .cmd_beats(rd_cmd_beats),
    .busy(dma_rd_busy), .done(rd_done_dma), .err(rd_err),
    .beat_valid, .beat_data, .beat_idx,
    .m_address(rd_address), .m_read(rd_read), .m_burstcount(rd_burstcount),
    .m_waitrequest(rd_waitrequest), .m_readdata(rd_readdata), .m_readdatavalid(rd_readdatavalid)
  );

  always_ff @(posedge clk) begin
    if (rd_cmd_valid && rd_cmd_ready) tgt_a <= rd_to_a;
  end

  wire [15:0] w_widx = beat_idx - 16'd8;     // weight beats follow the 8 constant beats
  wire a_we = beat_valid &&  tgt_a;
  wire w_we = beat_valid && !tgt_a && (beat_idx >= 16'd8);
  wire c_we = beat_valid && !tgt_a && (beat_idx <  16'd8);

  logic [127:0] a_rdata, w_rdata;
  logic [15:0]  fd_a_raddr, fd_w_raddr;
  logic signed [COLS-1:0][31:0] bias;
  logic        [COLS-1:0][30:0] m0;
  logic        [COLS-1:0][5:0]  shift;

  tile_buffers #(.A_BEATS(A_BEATS), .W_BEATS(W_BEATS)) u_buf (
    .clk,
    .a_we, .a_waddr(beat_idx[$clog2(A_BEATS)-1:0]), .a_wdata(beat_data),
    .a_raddr(fd_a_raddr[$clog2(A_BEATS):0]), .a_rdata,
    .w_load_sel, .w_we, .w_waddr(w_widx[$clog2(W_BEATS)-1:0]), .w_wdata(beat_data),
    .c_we, .c_idx(beat_idx[2:0]), .c_wdata(beat_data),
    .w_comp_sel, .w_raddr(fd_w_raddr[$clog2(W_BEATS):0]), .w_rdata,
    .bias, .m0, .shift
  );

  // ---- feeder and array ------------------------------------------------------------
  logic                              in_valid, in_first, in_shift, in_last, fd_busy;
  logic signed [ROWS-1:0][ACT_W-1:0] a_vec;
  logic signed [COLS-1:0][WGT_W-1:0] w_vec;

  operand_feeder #(.AW(16)) u_feed (
    .clk, .rst_n(rst_dp),
    .start(fd_start), .a_base(fd_a_base), .k_steps(fd_k), .two_pass(fd_two_pass), .in_signed(insign),
    .busy(fd_busy),
    .a_raddr(fd_a_raddr), .a_rdata, .w_raddr(fd_w_raddr), .w_rdata,
    .in_valid, .in_first, .in_shift, .in_last, .a_vec, .w_vec
  );

  logic                         row_valid, row_last;
  logic signed [COLS-1:0][15:0] row_data;
  logic signed [COLS-1:0][31:0] row_raw;

  linear_tile #(.LANES(LANES)) u_tile (
    .clk, .rst_n(rst_dp),
    .in_valid, .in_first, .in_shift, .in_last, .a_vec, .w_vec,
    .bias, .m0, .shift, .out16, .raw32,
    .busy(lt_busy), .row_valid, .row_data, .row_raw, .row_last
  );

  // ---- output rows: address, byte enables, FIFO ----------------------------------------
  logic [31:0] cur_addr, stride_q;
  logic [4:0]  rows_valid_q, rcnt;
  logic [31:0] cur_addr_next;

  always_ff @(posedge clk) begin
    row_last_seen <= 1'b0;
    if (!rst_dp) begin
      rcnt <= '0;
    end else begin
      if (sub_start) begin
        cur_addr     <= sub_addr;
        rows_valid_q <= sub_rows_valid;
        rcnt         <= '0;
        stride_q     <= stride_out;
      end else if (row_valid) begin
        cur_addr <= cur_addr + stride_q;
        rcnt     <= rcnt + 1'b1;
      end
      if (row_valid && row_last) row_last_seen <= 1'b1;
    end
  end

  logic [255:0] row_bus, row_lane;
  logic [31:0]  row_be;
  logic [127:0] row_bytes;
  always_comb begin
    for (int j = 0; j < COLS; j++) begin
      row_bus[16*j +: 16]  = row_data[j];
      row_bytes[8*j +: 8]  = row_data[j][7:0];
    end
    if (out16) begin
      row_lane = row_bus;
      row_be   = 32'hFFFF_FFFF;
    end else begin
      row_lane = cur_addr[4] ? {row_bytes, 128'b0} : {128'b0, row_bytes};
      row_be   = cur_addr[4] ? 32'hFFFF_0000 : 32'h0000_FFFF;
    end
  end

  logic         wr_in_valid, wr_in_ready, wr_busy, wr_beat, wr_err;
  wire push = row_valid && (rcnt < rows_valid_q);

  localparam int FW = $clog2(OUT_FIFO);
  logic [319:0] fifo_mem [OUT_FIFO];
  logic [FW:0]  wp, rp;
  wire fifo_empty = (wp == rp);
  wire fifo_full  = (wp[FW-1:0] == rp[FW-1:0]) && (wp[FW] != rp[FW]);
  logic         overflow;

  // raw INT32 rows are 64 bytes: the low half goes in with the push, the high half one cycle later
  // (rows come at least 4 cycles apart, so the FIFO write port is free)
  logic [255:0] raw_lo, raw_hi;
  logic         raw_hold;
  logic [319:0] raw_hold_entry;
  always_comb begin
    for (int j = 0; j < COLS / 2; j++) begin
      raw_lo[32*j +: 32] = row_raw[j];
      raw_hi[32*j +: 32] = row_raw[COLS/2 + j];
    end
  end

  always_ff @(posedge clk) begin
    overflow <= 1'b0;
    if (!rst_dp) begin
      wp <= '0;
      rp <= '0;
      raw_hold <= 1'b0;
    end else begin
      raw_hold <= push && raw32;
      if (push) begin
        if (fifo_full) overflow <= 1'b1;
        else begin
          fifo_mem[wp[FW-1:0]] <= raw32 ? {32'hFFFF_FFFF, {cur_addr[31:5], 5'b0}, raw_lo}
                                        : {row_be, {cur_addr[31:5], 5'b0}, row_lane};
          wp <= wp + 1'b1;
        end
        raw_hold_entry <= {32'hFFFF_FFFF, {cur_addr[31:5], 5'b0} + 32'd32, raw_hi};
      end else if (raw_hold) begin
        if (fifo_full) overflow <= 1'b1;
        else begin
          fifo_mem[wp[FW-1:0]] <= raw_hold_entry;
          wp <= wp + 1'b1;
        end
      end
      if (wr_in_valid && wr_in_ready) rp <= rp + 1'b1;
    end
  end

  logic [319:0] fifo_head;
  assign fifo_head   = fifo_mem[rp[FW-1:0]];
  assign wr_in_valid = !fifo_empty;

  dma_wr #(.TIMEOUT(DMA_TIMEOUT)) u_dma_wr (
    .clk, .rst_n(rst_dp),
    .in_valid(wr_in_valid), .in_ready(wr_in_ready),
    .in_addr(fifo_head[287:256]), .in_data(fifo_head[255:0]), .in_be(fifo_head[319:288]),
    .busy(wr_busy), .wr_beat, .err(wr_err),
    .m_address(wr_address), .m_write(wr_write), .m_writedata(wr_writedata),
    .m_byteenable(wr_byteenable), .m_burstcount(wr_burstcount), .m_waitrequest(wr_waitrequest)
  );

  assign wr_idle = fifo_empty && !wr_busy;

  // ---- errors, progress, counters ----------------------------------------------------------
  always_comb begin
    ext_err_valid = wr_err || overflow;
    ext_err_code  = wr_err ? 8'd4 : 8'd5;
  end
  assign progress = beat_valid || wr_beat || in_valid || push;

  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      bytes_rd <= '0;
      bytes_wr <= '0;
    end else begin
      if (beat_valid) bytes_rd <= bytes_rd + 32'd32;
      if (wr_beat)    bytes_wr <= bytes_wr + ((out16 || raw32) ? 32'd32 : 32'd16);
    end
  end
endmodule
