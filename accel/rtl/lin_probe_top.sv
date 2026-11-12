// Resource probe for linear_tile (array + epilogue): pseudo-random stimulus from an
// internal LFSR, outputs folded into 8 pins. Only for Quartus resource/timing numbers;
// not part of the final design.
module lin_probe_top
  import fpgate_pkg::*;
#(
  parameter int LANES = 4
)(
  input  logic       clk,
  input  logic       rst_n,
  output logic [7:0] led
);
  localparam int NA = PE_ROWS * ACT_W;
  localparam int NW = PE_COLS * WGT_W;
  localparam int NB = PE_COLS * 32;
  localparam int NM = PE_COLS * 31;
  localparam int NS = PE_COLS * 6;
  localparam int N  = NA + NW + NB + NM + NS;
  localparam int K  = 64;

  logic [N-1:0] lfsr;
  logic [9:0]   cnt;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      lfsr <= N'(1);
      cnt  <= '0;
    end else begin
      lfsr <= {lfsr[N-2:0], lfsr[N-1] ^ lfsr[N-2] ^ lfsr[N-5] ^ lfsr[N-17]};
      cnt  <= cnt + 1'b1;
    end
  end

  logic                              busy, row_valid, row_last;
  logic signed [PE_COLS-1:0][15:0]   row_data;
  logic [7:0]                        acc8;

  linear_tile #(.LANES(LANES)) u_tile (
    .clk, .rst_n,
    .in_valid(cnt < K),
    .in_first(cnt == 0),
    .in_shift(1'b0),
    .in_last (cnt == K-1),
    .in_tag(1'b0), .last_ok(), .drain_tag(), .out_room(1'b1),
    .a_vec(lfsr[NA-1:0]),
    .w_vec(lfsr[NA +: NW]),
    .bias (lfsr[NA+NW +: NB]),
    .m0   (lfsr[NA+NW+NB +: NM]),
    .shift(lfsr[NA+NW+NB+NM +: NS]),
    .out16(lfsr[3]), .raw32(1'b0),
    .busy, .row_valid, .row_data, .row_raw(), .row_last
  );

  always_ff @(posedge clk) begin
    if (row_valid) begin
      logic [7:0] x;
      x = '0;
      for (int j = 0; j < PE_COLS; j++) x ^= row_data[j][7:0] ^ row_data[j][15:8];
      acc8 <= acc8 ^ x;
    end
    led <= acc8;
  end
endmodule
