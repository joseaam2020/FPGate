// One 16x16 output tile of a linear layer: GEMM array + epilogue.
// Feed K k-steps (two passes for 16-bit inputs, see gemm_array), then receive
// ROWS requantized rows on row_*. The epilogue back-pressures the array's drain.
// Tiles may be fed back to back: the last k-step of a tile is only presented while
// last_ok is high (see gemm_array). in_tag goes with the tile and selects, on
// drain_tag, the set of channel constants the epilogue must see for its rows.
module linear_tile
  import fpgate_pkg::*;
#(
  parameter int ROWS  = PE_ROWS,
  parameter int COLS  = PE_COLS,
  parameter int LANES = 4
)(
  input  logic                              clk,
  input  logic                              rst_n,
  // GEMM side
  input  logic                              in_valid,
  input  logic                              in_first,
  input  logic                              in_shift,
  input  logic                              in_last,
  input  logic                              in_tag,
  input  logic signed [ROWS-1:0][ACT_W-1:0] a_vec,
  input  logic signed [COLS-1:0][WGT_W-1:0] w_vec,
  output logic                              last_ok,
  output logic                              drain_tag,  // tag of the tile whose rows enter the epilogue
  // per-tile channel constants (of the tile on drain_tag)
  input  logic signed [COLS-1:0][31:0]      bias,
  input  logic        [COLS-1:0][30:0]      m0,
  input  logic        [COLS-1:0][5:0]       shift,
  input  logic                              out16,
  input  logic                              raw32,      // attention scores: raw INT32 rows on row_raw
  input  logic                              out_room,   // the consumer can take one more row (registered)
  // result
  output logic                              busy,
  output logic                              row_valid,
  output logic signed [COLS-1:0][15:0]      row_data,
  output logic signed [COLS-1:0][31:0]      row_raw,
  output logic                              row_last
);
  logic                              d_valid, d_ready, d_last;
  logic [$clog2(ROWS)-1:0]           d_row;
  logic signed [COLS-1:0][ACC_W-1:0] d_data;
  logic                              arr_busy;
  logic [1:0]                        n_ep;      // tiles fed whose last requantized row is not out yet

  gemm_array #(.ROWS(ROWS), .COLS(COLS)) u_array (
    .clk, .rst_n,
    .in_valid, .in_first, .in_shift, .in_last, .in_tag, .a_vec, .w_vec,
    .busy(arr_busy), .last_ok, .drain_tag,
    .drain_valid(d_valid), .drain_ready(d_ready), .drain_last(d_last),
    .drain_row(d_row), .drain_data(d_data)
  );

  epilogue_row #(.COLS(COLS), .LANES(LANES)) u_ep (
    .clk, .rst_n,
    .bias, .m0, .shift, .out16, .raw(raw32), .out_room,
    .in_valid(d_valid), .in_ready(d_ready), .in_data(d_data), .in_last(d_last),
    .row_valid, .row_data, .row_raw, .row_last
  );

  // busy until the last requantized row of the last tile fed is out
  always_ff @(posedge clk) begin
    if (!rst_n) n_ep <= '0;
    else        n_ep <= n_ep + 2'(in_valid & in_last) - 2'(row_valid & row_last);
  end
  assign busy = arr_busy | (n_ep != 2'd0);
endmodule
