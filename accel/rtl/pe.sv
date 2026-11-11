// Processing element: 9x8 signed multiplier plus a 32-bit accumulator.
// Output-stationary: the activation flows right, the weight flows down, and
// the control tags (valid/first/shift) travel with the activation.
//
// 16-bit inputs take two passes (high byte, then low byte). On the first
// product of the second pass `shift` is set and the accumulator is moved
// left 8 bits before adding: acc = 256*sum(hi*w) + sum(lo*w).
module pe #(
  parameter int ACC_W = 32
)(
  input  logic                     clk,
  input  logic                     rst_n,
  input  logic signed [8:0]        a_in,
  input  logic                     a_valid_in,
  input  logic                     a_first_in,
  input  logic                     a_shift_in,
  output logic signed [8:0]        a_out,
  output logic                     a_valid_out,
  output logic                     a_first_out,
  output logic                     a_shift_out,
  input  logic signed [7:0]        w_in,
  output logic signed [7:0]        w_out,
  output logic signed [ACC_W-1:0]  acc
);
  logic signed [16:0] prod;
  // dont_merge: identical tags in PEs on the same anti-diagonal must stay separate
  // registers, otherwise one register fans out to distant PEs and breaks timing.
  (* dont_merge *) logic valid_d;
  (* dont_merge *) logic first_d;
  (* dont_merge *) logic shift_d;

  always_ff @(posedge clk) begin
    // registered propagation (no reset to save logic)
    a_out       <= a_in;
    a_first_out <= a_first_in;
    a_shift_out <= a_shift_in;
    w_out       <= w_in;
    // stage 1: multiply
    prod    <= a_in * w_in;
    first_d <= a_first_in;
    shift_d <= a_shift_in;
  end

  // valid tags are reset so the accumulator never sees an X after reset
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      a_valid_out <= 1'b0;
      valid_d     <= 1'b0;
    end else begin
      a_valid_out <= a_valid_in;
      valid_d     <= a_valid_in;
    end
  end

  // stage 2: accumulate
  always_ff @(posedge clk) begin
    if (!rst_n)
      acc <= '0;
    else if (valid_d)
      acc <= first_d ? ACC_W'(prod)
           : shift_d ? (acc <<< 8) + ACC_W'(prod)
           :           acc + ACC_W'(prod);
  end
endmodule
