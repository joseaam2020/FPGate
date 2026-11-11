// Resource probe: gemm_array fed by an internal pseudo-random generator, with
// the whole drain output folded into 8 pins. Exists only so Quartus can place
// and route the array (few pins) and report real DSP/ALM/timing numbers.
// Not part of the final design.
module dsp_probe_top
  import fpgate_pkg::*;
(
  input  logic       clk,
  input  logic       rst_n,
  output logic [7:0] led
);
  localparam int NA = PE_ROWS * ACT_W;
  localparam int NW = PE_COLS * WGT_W;
  localparam int N  = NA + NW;
  localparam int K  = 64;

  logic [7:0]   led_a, led_b;
  logic [N-1:0] lfsr;
  logic [7:0]   cnt;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      lfsr <= N'(1);
      cnt  <= '0;
    end else begin
      lfsr <= {lfsr[N-2:0], lfsr[N-1] ^ lfsr[N-2] ^ lfsr[N-5] ^ lfsr[N-17]};
      cnt  <= cnt + 1'b1;
    end
  end

  logic                              busy, drain_valid;
  logic [$clog2(PE_ROWS)-1:0]        drain_row;
  logic signed [PE_COLS-1:0][ACC_W-1:0] drain_data;

  gemm_array u_array (
    .clk, .rst_n,
    .in_valid(cnt < K),
    .in_first(cnt == 0),
    .in_shift(1'b0),
    .in_last (cnt == K-1),
    .a_vec(lfsr[NA-1:0]),
    .w_vec(lfsr[N-1:NA]),
    .busy, .drain_valid, .drain_ready(1'b1), .drain_last(), .drain_row, .drain_data
  );

  always_ff @(posedge clk) begin
    if (drain_valid) begin
      logic [7:0] x;
      x = '0;
      for (int j = 0; j < PE_COLS; j++)
        x ^= drain_data[j][7:0] ^ drain_data[j][15:8] ^ drain_data[j][23:16] ^ drain_data[j][31:24];
      led_a <= led_a ^ x;
    end
  end
  assign led = led_a ^ led_b;
  // Epilogue-like load: 4 full 32x31 products (acc * M0), to check DSP headroom.
  logic signed [31:0] ea [4];
  logic signed [30:0] eb [4];
  logic signed [62:0] ep [4];
  always_ff @(posedge clk) begin
    for (int e = 0; e < 4; e++) begin
      ea[e] <= lfsr[32*e +: 32];
      eb[e] <= lfsr[31*e+8 +: 31];
      ep[e] <= ea[e] * eb[e];
    end
    led_b <= led_b ^ ep[0][62:55] ^ ep[1][62:55] ^ ep[2][62:55] ^ ep[3][62:55];
  end
endmodule
