// One lane of the softmax normalisation (pass 3 of the integer softmax):
//
//   p = min(65535, (e * R + 2^(sh-1)) >> sh)         e UQ0.16 exponential, R reciprocal table value (< 2^15)
//
// with sh = 15 + k - 16 (k = position of the leading one of the row sum, 14 <= sh <= 25). The 16 x 15 product is
// built from two small multipliers in logic (no DSP block). Fixed latency P3_LATENCY = 5 cycles.
(* multstyle = "logic" *)
module softmax_p3_lane (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [15:0] e,
  input  logic [14:0] r,
  input  logic [4:0]  sh,
  input  logic [25:0] rnd,            // 2^(sh - 1)
  output logic        out_valid,
  output logic [15:0] p
);
  logic        v0, v1, v2, vi;
  logic [15:0] e_i;
  logic [14:0] r_i;
  logic [23:0] lo;
  logic [22:0] hi;
  logic [31:0] prod, x;

  always_ff @(posedge clk) begin
    vi  <= rst_n & in_valid;          // input register: e comes straight from a RAM output
    e_i <= e;
    r_i <= r;
    v0  <= vi & rst_n;
    lo <= e_i * r_i[7:0];
    hi <= e_i * r_i[14:8];
    v1 <= v0 & rst_n;
    prod <= {8'b0, lo} + {1'b0, hi, 8'b0} + {6'b0, rnd};
    v2 <= v1 & rst_n;
    x <= prod >> sh;
    out_valid <= v2 & rst_n;
    p <= (x > 32'd65535) ? 16'hFFFF : x[15:0];
  end
endmodule
