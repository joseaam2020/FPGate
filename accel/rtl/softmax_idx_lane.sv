// One lane of the softmax exponent index (pass 2 of the integer softmax):
//
//   idx = min(1024, ((max - s) * M0 + 2^(sh-1)) >> sh)          (M0 31 bit, sh 1..46)
//
// d = max - s is non-negative and below 2^21 for the model's scores (|S| <= 127 * 127 * 64 < 2^20); a larger
// value (corrupt data) saturates the index. The product is built from radix-4 partial products and a
// pipelined adder tree, with no DSP block (the engine uses all of them): 11 digits of d times {0, M0, 2*M0,
// 3*M0}, plus the rounding term, summed in 4 levels. Fixed latency IDX_LATENCY cycles.
(* multstyle = "logic" *)
module softmax_idx_lane
#(
  parameter int DW = 21                     // bits of d = max - s that take part in the product
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               in_valid,
  input  logic signed [31:0] s,
  input  logic signed [31:0] mx,
  // per operation constants (stable while the lane works)
  input  logic        [30:0] m0,
  input  logic        [32:0] m0x3,          // 3 * M0
  input  logic        [5:0]  sh,
  input  logic        [53:0] rnd,           // 2^(sh - 1)
  output logic               out_valid,
  output logic        [10:0] idx
);
  localparam int ND = (DW + 1) / 2;          // radix-4 digits (11)

  // stage 0: d = max - s, saturation flag
  logic               v0;
  logic [2*ND-1:0]    d0;                // even width: the last radix-4 digit reads two bits
  logic               big0;
  logic signed [32:0] diff;
  assign diff = {mx[31], mx} - {s[31], s};
  always_ff @(posedge clk) begin
    v0   <= rst_n & in_valid;
    d0   <= (2*ND)'(diff[DW-1:0]);
    big0 <= diff[32] | (|diff[31:DW]);
  end

  // stage 1: partial products
  logic          v1, big1;
  logic [53:0]   pp [ND+1];
  always_ff @(posedge clk) begin
    v1   <= v0 & rst_n;
    big1 <= big0;
    for (int i = 0; i < ND; i++) begin
      logic [1:0]  dg;
      logic [34:0] sel;
      dg = d0[2*i +: 2];
      case (dg)
        2'd0:    sel = '0;
        2'd1:    sel = {4'b0, m0};
        2'd2:    sel = {3'b0, m0, 1'b0};
        default: sel = {2'b0, m0x3};
      endcase
      pp[i] <= 54'(sel) << (2 * i);
    end
    pp[ND] <= rnd;
  end

  // stages 2..5: adder tree, 12 -> 6 -> 3 -> 2 -> 1
  logic        v2, v3, v4, v5, big2, big3, big4, big5;
  logic [53:0] t [6];
  logic [53:0] u [3];
  logic [53:0] w [2];
  logic [53:0] prod;
  always_ff @(posedge clk) begin
    v2 <= v1 & rst_n; big2 <= big1;
    for (int i = 0; i < 6; i++) t[i] <= pp[2*i] + pp[2*i+1];
    v3 <= v2 & rst_n; big3 <= big2;
    for (int i = 0; i < 3; i++) u[i] <= t[2*i] + t[2*i+1];
    v4 <= v3 & rst_n; big4 <= big3;
    w[0] <= u[0] + u[1];
    w[1] <= u[2];
    v5 <= v4 & rst_n; big5 <= big4;
    prod <= w[0] + w[1];
  end

  // stages 6, 7: arithmetic right shift by sh (barrel, in two steps), then clamp
  logic        v6, v7, big6, big7;
  logic [53:0] x1, x2;
  always_ff @(posedge clk) begin
    v6 <= v5 & rst_n; big6 <= big5;
    x1 <= prod >> {sh[5:3], 3'b000};
    v7 <= v6 & rst_n; big7 <= big6;
    x2 <= x1 >> sh[2:0];
    out_valid <= v7 & rst_n;
    idx <= (big7 || x2 > 54'd1024) ? 11'd1024 : x2[10:0];
  end
endmodule
