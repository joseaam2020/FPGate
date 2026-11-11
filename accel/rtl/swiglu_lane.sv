// SwiGLU lane of the epilogue, bit-exact with export_check.run:
//   sg = silu_table[gi + 127]          (255 entries, INT16, one table per block)
//   h  = sat16( (sg * ui * M0 + 2^(shift-1)) >>> shift )
// gi/ui are the already requantized INT8 outputs of fc1g/fc1x. M0/shift are scalars
// of the block (manifest: swiglu_requant). The table is loaded through the write
// port before the block starts. The final requantization reuses requant_lane.
// Latency: 3 + requant_lane (6) cycles, one value per cycle.
module swiglu_lane (
  input  logic                clk,
  input  logic                rst_n,
  // table load (one entry per cycle)
  input  logic                tbl_we,
  input  logic        [7:0]   tbl_addr,
  input  logic signed [15:0]  tbl_data,
  // data
  input  logic                in_valid,
  input  logic signed [7:0]   gi,
  input  logic signed [7:0]   ui,
  input  logic        [30:0]  m0,
  input  logic        [5:0]   shift,
  output logic                out_valid,
  output logic signed [15:0]  h
);
  logic signed [15:0] tbl [256];

  always_ff @(posedge clk) if (tbl_we) tbl[tbl_addr] <= tbl_data;

  // stage 1: table address
  logic        [7:0]  addr_1;
  logic signed [7:0]  ui_1;
  logic        [30:0] m0_1;
  logic        [5:0]  sh_1;
  logic               v_1;
  // stage 2: table read
  logic signed [15:0] sg_2;
  logic signed [7:0]  ui_2;
  logic        [30:0] m0_2;
  logic        [5:0]  sh_2;
  logic               v_2;
  // stage 3: product
  logic signed [23:0] prod_3;
  logic        [30:0] m0_3;
  logic        [5:0]  sh_3;
  logic               v_3;

  always_ff @(posedge clk) begin
    addr_1 <= 8'(gi + 8'sd127);   // 0..254 for gi in [-127, 127]
    ui_1   <= ui;
    m0_1   <= m0;
    sh_1   <= shift;

    sg_2   <= tbl[addr_1];        // synchronous read: infers M10K
    ui_2   <= ui_1;
    m0_2   <= m0_1;
    sh_2   <= sh_1;

    prod_3 <= sg_2 * ui_2;
    m0_3   <= m0_2;
    sh_3   <= sh_2;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v_1 <= 1'b0; v_2 <= 1'b0; v_3 <= 1'b0;
    end else begin
      v_1 <= in_valid; v_2 <= v_1; v_3 <= v_2;
    end
  end

  requant_lane u_rq (
    .clk, .rst_n,
    .in_valid(v_3),
    .acc(32'(prod_3)),
    .bias(32'sd0),
    .m0(m0_3),
    .shift(sh_3),
    .out16(1'b1),
    .out_valid,
    .y(h)
  );
endmodule
