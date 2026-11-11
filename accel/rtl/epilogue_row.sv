// Linear-layer epilogue for one row of COLS accumulators, with LANES parallel
// requant_lane instances. A row is processed in COLS/LANES beats; the requantized
// values are collected and delivered as a full row (row_valid, row_data, row_last).
//
// Per-channel constants (bias, M0, shift) belong to the 16 output channels of the
// current tile and must stay stable while the tile is in flight.
// Input handshake: in_valid && in_ready accepts a row. At LANES = COLS a row is
// accepted every cycle; at LANES = 4 every 4 cycles.
module epilogue_row
  import fpgate_pkg::*;
#(
  parameter int COLS  = 16,
  parameter int LANES = 4
)(
  input  logic                          clk,
  input  logic                          rst_n,
  // per-tile, per-channel constants
  input  logic signed [COLS-1:0][31:0]  bias,
  input  logic        [COLS-1:0][30:0]  m0,
  input  logic        [COLS-1:0][5:0]   shift,
  input  logic                          out16,
  input  logic                          raw,        // deliver the accumulator row unchanged (row_raw)
  // accumulator rows
  input  logic                          in_valid,
  output logic                          in_ready,
  input  logic signed [COLS-1:0][31:0]  in_data,
  input  logic                          in_last,
  // requantized rows
  output logic                          row_valid,
  output logic signed [COLS-1:0][15:0]  row_data,
  output logic signed [COLS-1:0][31:0]  row_raw,    // valid with row_valid when raw = 1
  output logic                          row_last
);
  localparam int NB      = COLS / LANES;     // beats per row
  localparam int RQ_LAT  = RQ_LATENCY;       // requant_lane latency (see fpgate_pkg)
  localparam int BW      = (NB > 1) ? $clog2(NB) : 1;

  // The array holds the row on in_data/in_last while in_ready is low, so the
  // epilogue reads it in place (no copy) and only releases it on the last beat.
  // in_ready is a register that is high exactly during the last beat.
  logic [BW-1:0] beat, beat_n;
  logic          ready_q;
  assign in_ready = ready_q;

  wire accept = in_valid && ready_q;

  always_comb begin
    beat_n = beat;
    if (accept)        beat_n = '0;
    else if (in_valid) beat_n = beat + 1'b1;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      beat    <= '0;
      ready_q <= (NB == 1);
    end else begin
      beat    <= beat_n;
      ready_q <= (beat_n == BW'(NB - 1));
    end
  end

  // column select registered once before the lanes (the 16:1 selection of acc,
  // bias, M0 and shift is too slow to share a cycle with the bias adder)
  logic                lane_valid;
  logic signed [31:0]  l_acc   [LANES];
  logic signed [31:0]  l_bias  [LANES];
  logic        [30:0]  l_m0    [LANES];
  logic        [5:0]   l_sh    [LANES];
  logic [BW-1:0]       beat_r;
  logic                lastbeat_r, rowlast_r;

  always_ff @(posedge clk) begin
    for (int l = 0; l < LANES; l++) begin
      l_acc[l]  <= in_data[beat * LANES + l];
      l_bias[l] <= bias[beat * LANES + l];
      l_m0[l]   <= m0[beat * LANES + l];
      l_sh[l]   <= shift[beat * LANES + l];
    end
    beat_r     <= beat;
    lastbeat_r <= (beat == BW'(NB - 1));
    rowlast_r  <= in_last;
    lane_valid <= rst_n & in_valid & ~raw;
  end

  // ---- lanes ------------------------------------------------------------------
  logic                lane_out_valid [LANES];
  logic signed [15:0]  lane_y         [LANES];

  genvar gl;
  generate
  for (gl = 0; gl < LANES; gl++) begin : g_lane
    requant_lane u_lane (
      .clk, .rst_n,
      .in_valid(lane_valid),
      .acc(l_acc[gl]), .bias(l_bias[gl]), .m0(l_m0[gl]), .shift(l_sh[gl]), .out16,
      .out_valid(lane_out_valid[gl]), .y(lane_y[gl])
    );
  end
  endgenerate

  // sideband (beat index, last beat of the last row) follows the lanes' latency
  logic [BW-1:0] beat_d [RQ_LAT];
  logic          lastbeat_d [RQ_LAT];
  logic          rowlast_d [RQ_LAT];
  always_ff @(posedge clk) begin
    beat_d[0]     <= beat_r;
    lastbeat_d[0] <= lastbeat_r;
    rowlast_d[0]  <= rowlast_r;
    for (int d = 1; d < RQ_LAT; d++) begin
      beat_d[d]     <= beat_d[d-1];
      lastbeat_d[d] <= lastbeat_d[d-1];
      rowlast_d[d]  <= rowlast_d[d-1];
    end
  end

  // ---- collect ----------------------------------------------------------------
  logic signed [COLS-1:0][15:0] acc_row;

  always_ff @(posedge clk) begin
    row_valid <= 1'b0;
    if (!rst_n) begin
      row_last <= 1'b0;
    end else if (raw) begin
      // raw mode: capture the accumulator row when it is accepted (the array holds it until then)
      if (accept) begin
        row_raw   <= in_data;
        row_valid <= 1'b1;
        row_last  <= in_last;
      end
    end else if (lane_out_valid[0]) begin
      logic signed [COLS-1:0][15:0] nxt;
      nxt = acc_row;
      for (int l = 0; l < LANES; l++) nxt[beat_d[RQ_LAT-1] * LANES + l] = lane_y[l];
      acc_row <= nxt;
      if (lastbeat_d[RQ_LAT-1]) begin
        row_data  <= nxt;
        row_valid <= 1'b1;
        row_last  <= rowlast_d[RQ_LAT-1];
      end
    end
  end
endmodule
