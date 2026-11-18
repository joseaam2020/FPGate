// RoPE on the INT16 output rows of a q / k projection (bit-exact rope_int_range of the software:
// interleaved pairs, Q1.14 table, one rounding shift, saturation to +-127).
//
// Rows of 16 INT16 channels (one tile of a head) come in at least NB = 4 cycles apart (the epilogue
// delivers a row every COLS / LANES cycles) and leave as rows of 16 INT8 values (sign-extended to 16 bits,
// so the engine's INT8 output path takes them unchanged). A row takes 4 beats of 4 channels (two pairs); a
// pair costs four 16x16 products, so 8 multipliers are enough.
//
//   v(2i)   = x(2i)   * c - x(2i+1) * s        (c = cos, s = sin of pair i, token t)
//   v(2i+1) = x(2i+1) * c + x(2i)   * s
//   y = sat8((v + 2^(sh-1)) >>> sh)             sh = frac + pre_shift
//   token 0 (class token) is not rotated: y = sat8((x + 2^(pre-1)) >>> pre), which is the same formula with
//   c = 2^(sh-pre) and s = 0.
//
// The r7 tables are separable (tools/gen_rope_tables.py checks it on the real tables.bin): for the token
// t = 1 + 32*y + x and the head dimension d,   d < 32: depends on y only;   d >= 32: on x only;   the
// values of the pair (2i, 2i+1) are identical. Four RAMs of 512 x 16 bit (cos / sin, pair lane 0 / 1), indexed
//   {half = d[5], yx (5 bit: y when half = 0, x when half = 1), pair-in-half (4 bit)}
// replace the 1024 x 64 x 2 table (8 KiB instead of 256 KiB). A row is one tile of 16 channels: th = tile
// within the head (hd = 64: 0..3), half = th[1], pair-in-half = {th[0], beat, lane}.
//
// Table load: tab_we with tab_data = {5'b0, sin, pidx[9:0], value[15:0]}, pidx = {half, yx, pair-in-half}.
// The sideband of a subtile (first token of its row 0, tile-in-head) is pushed with ctx_push when the
// sequencer starts the subtile; the unit takes it back when the subtile's last row has been processed.
// Multipliers in logic: the DSP blocks are all taken by the GEMM array and the epilogue.
(* multstyle = "logic" *) module rope_unit #(
  parameter int COLS = 16
)(
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          en,
  input  logic [5:0]                    sh,          // frac + pre_shift
  input  logic [5:0]                    pre,         // pre_shift (token 0)
  // table write port
  input  logic                          tab_we,
  input  logic [31:0]                   tab_data,
  // subtile context, pushed at sub_start
  input  logic                          ctx_push,
  input  logic [15:0]                   ctx_row0,
  input  logic [1:0]                    ctx_th,
  // INT16 rows in
  input  logic                          in_valid,
  input  logic signed [COLS-1:0][15:0]  in_data,
  input  logic                          in_last,
  // INT8 rows out (sign-extended)
  output logic                          out_valid,
  output logic signed [COLS-1:0][15:0]  out_data,
  output logic                          out_last,
  output logic                          busy
);
  // ---- tables --------------------------------------------------------------------
  // four simple dual-port RAMs (cos / sin x pair lane 0 / 1), 512 x 16 bit each: one M10K each
  logic signed [15:0] cos_q [2];
  logic signed [15:0] sin_q [2];
  logic [8:0]         raddr;

  wire        tab_sin = tab_data[26];
  wire  [9:0] tab_idx = tab_data[25:16];
  wire  [8:0] tab_adr = {tab_idx[9:4], tab_idx[3:1]};   // {half, yx, pair-in-half[3:1]}
  wire        tab_ln  = tab_idx[0];

  genvar gl;
  generate
    for (gl = 0; gl < 2; gl++) begin : g_tab
      logic [15:0] cos_ram [512];
      logic [15:0] sin_ram [512];
      always_ff @(posedge clk) begin
        if (tab_we && !tab_sin && tab_ln == gl) cos_ram[tab_adr] <= tab_data[15:0];
        if (tab_we &&  tab_sin && tab_ln == gl) sin_ram[tab_adr] <= tab_data[15:0];
        cos_q[gl] <= cos_ram[raddr];
        sin_q[gl] <= sin_ram[raddr];
      end
    end
  endgenerate

  // ---- subtile contexts ------------------------------------------------------------
  logic [15:0] cx_row0 [4];
  logic [1:0]  cx_th   [4];
  logic [2:0]  cx_wp, cx_rp;
  logic [3:0]  ri;                     // row within the subtile
  wire         cx_have = (cx_wp != cx_rp);

  // ---- row capture -----------------------------------------------------------------
  logic signed [COLS-1:0][15:0] hold;
  logic        active, last_q, is0_q;
  logic [1:0]  bc;
  logic [1:0]  th_q;
  logic [9:0]  u_q;                    // token - 1

  wire [15:0] tok = cx_row0[cx_rp[1:0]] + 16'(ri);
  wire        take = en && in_valid;

  always_ff @(posedge clk) begin
    if (!rst_n || !en) begin
      cx_wp <= '0; cx_rp <= '0; ri <= '0; active <= 1'b0; bc <= '0;
    end else begin
      if (ctx_push) begin
        cx_row0[cx_wp[1:0]] <= ctx_row0;
        cx_th[cx_wp[1:0]]   <= ctx_th;
        cx_wp               <= cx_wp + 1'b1;
      end
      if (take) begin
        hold   <= in_data;
        last_q <= in_last;
        is0_q  <= (tok == 16'd0);
        u_q    <= 10'(tok - 16'd1);
        th_q   <= cx_th[cx_rp[1:0]];
        active <= 1'b1;
        bc     <= '0;
        if (in_last) begin ri <= '0; cx_rp <= cx_rp + 1'b1; end
        else         ri <= ri + 1'b1;
      end else if (active) begin
        bc <= bc + 1'b1;
        if (bc == 2'd3) active <= 1'b0;
      end
    end
  end
  // ---- stage A: table address, channel select ------------------------------------------
  logic signed [15:0] xa [4];
  logic        v_a, is0_a, last_a;
  logic [1:0]  bc_a;
  wire         half = th_q[1];

  always_ff @(posedge clk) begin
    raddr  <= {half, half ? u_q[4:0] : u_q[9:5], th_q[0], bc};
    for (int i = 0; i < 4; i++) xa[i] <= hold[{bc, 2'b00} + 4'(i)];
    v_a    <= rst_n & en & active;
    is0_a  <= is0_q;
    last_a <= last_q && (bc == 2'd3);
    bc_a   <= bc;
  end

  // ---- stage B: the operands wait one cycle for the RAM (raddr is registered in A, the RAM output in B) ----
  logic signed [15:0] xb [4];
  logic        v_b, is0_b, last_b;
  logic [1:0]  bc_b;
  logic signed [15:0] c_b [2];
  logic signed [15:0] s_b [2];
  logic signed [31:0] m_xc0 [2], m_x1s [2], m_x1c [2], m_x0s [2];

  wire signed [15:0] c0 = 16'(32'sd1 <<< (sh - pre));

  always_ff @(posedge clk) begin
    for (int i = 0; i < 4; i++) xb[i] <= xa[i];
    v_b    <= rst_n & en & v_a;
    is0_b  <= is0_a;
    last_b <= last_a;
    bc_b   <= bc_a;
  end

  // ---- stage C: products (the RAM data are available together with xb) ----
  logic        v_c, is0_c, last_c;
  logic [1:0]  bc_c;
  always_comb begin
    for (int l = 0; l < 2; l++) begin
      c_b[l] = is0_b ? c0 : cos_q[l];
      s_b[l] = is0_b ? 16'sd0 : sin_q[l];
    end
  end
  always_ff @(posedge clk) begin
    for (int l = 0; l < 2; l++) begin
      m_xc0[l] <= xb[2*l]     * c_b[l];
      m_x1s[l] <= xb[2*l + 1] * s_b[l];
      m_x1c[l] <= xb[2*l + 1] * c_b[l];
      m_x0s[l] <= xb[2*l]     * s_b[l];
    end
    v_c    <= rst_n & en & v_b;
    last_c <= last_b;
    bc_c   <= bc_b;
  end

  // ---- stage D: sum with the rounding constant, then shift and saturate ----------------------
  logic signed [33:0] v0 [2], v1 [2];
  logic        v_d, last_d;
  logic [1:0]  bc_d;
  wire signed [33:0] rnd = 34'sd1 <<< (sh - 6'd1);

  always_ff @(posedge clk) begin
    for (int l = 0; l < 2; l++) begin
      v0[l] <= 34'(m_xc0[l]) - 34'(m_x1s[l]) + rnd;
      v1[l] <= 34'(m_x1c[l]) + 34'(m_x0s[l]) + rnd;
    end
    v_d    <= rst_n & en & v_c;
    last_d <= last_c;
    bc_d   <= bc_c;
  end

  // ---- stage E: shift, saturate, collect ---------------------------------------------------------
  function automatic logic signed [15:0] sat8(input logic signed [33:0] v, input logic [5:0] s);
    logic signed [33:0] r;
    r = v >>> s;
    if (r > 34'sd127)       return 16'sd127;
    else if (r < -34'sd127) return -16'sd127;
    else                    return 16'(r);
  endfunction

  logic signed [COLS-1:0][15:0] acc_row;
  assign busy = active || v_a || v_b || v_c || v_d;
  always_ff @(posedge clk) begin
    out_valid <= 1'b0;
    if (v_d) begin
      logic signed [COLS-1:0][15:0] nxt;
      nxt = acc_row;
      for (int l = 0; l < 2; l++) begin
        nxt[{bc_d, 2'b00} + 4'(2*l)]     = sat8(v0[l], sh);
        nxt[{bc_d, 2'b00} + 4'(2*l + 1)] = sat8(v1[l], sh);
      end
      acc_row <= nxt;
      if (bc_d == 2'd3) begin
        out_data  <= nxt;
        out_valid <= rst_n & en;
        out_last  <= last_d;
      end
    end
  end
endmodule
