// Feeds the GEMM array from the on-chip RAMs. For a subtile it walks K steps (twice for
// 16-bit inputs: plane 0 = high byte, plane 1 = low byte) and drives the array's
// per-cycle operands. RAMs have a 1-cycle read latency, so the control flags are
// delayed one cycle to line up with the data.
//
// Byte mapping into the array's 9-bit activation input:
//   8-bit input          : sign-extended byte
//   16-bit, plane 0 (hi) : signed if in_signed, else zero-extended
//   16-bit, plane 1 (lo) : zero-extended
module operand_feeder
  import fpgate_pkg::*;
#(
  parameter int ROWS = PE_ROWS,
  parameter int COLS = PE_COLS,
  parameter int AW   = 16
)(
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              start,       // 1-cycle pulse
  input  logic [AW-1:0]                     a_base,      // word index of plane 0, k = 0 of this subtile
  input  logic [AW-1:0]                     k_steps,     // K
  input  logic                              two_pass,
  input  logic                              in_signed,
  output logic                              busy,
  // RAM read ports
  output logic [AW-1:0]                     a_raddr,
  input  logic [127:0]                      a_rdata,
  output logic [AW-1:0]                     w_raddr,
  input  logic [127:0]                      w_rdata,
  // towards linear_tile
  output logic                              in_valid,
  output logic                              in_first,
  output logic                              in_shift,
  output logic                              in_last,
  output logic signed [ROWS-1:0][ACT_W-1:0] a_vec,
  output logic signed [COLS-1:0][WGT_W-1:0] w_vec
);
  logic [AW-1:0] k;
  logic          pass, run;
  logic [AW-1:0] kmax;

  assign busy    = run;
  assign a_raddr = a_base + (pass ? k_steps : '0) + k;
  assign w_raddr = k;

  // stage-1 control (aligned with the RAM outputs)
  logic v1, f1, s1, l1, p1;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      run <= 1'b0; pass <= 1'b0; k <= '0;
      v1 <= 1'b0; f1 <= 1'b0; s1 <= 1'b0; l1 <= 1'b0; p1 <= 1'b0;
    end else begin
      v1 <= run;
      f1 <= run && (k == 0) && !pass;
      s1 <= run && (k == 0) &&  pass;
      l1 <= run && (k == k_steps - 1) && (pass == two_pass);
      p1 <= pass;
      if (start) begin
        run  <= 1'b1;
        pass <= 1'b0;
        k    <= '0;
      end else if (run) begin
        if (k == k_steps - 1) begin
          k <= '0;
          if (two_pass && !pass) pass <= 1'b1;
          else                   run  <= 1'b0;
        end else begin
          k <= k + 1'b1;
        end
      end
    end
  end

  assign in_valid = v1;
  assign in_first = f1;
  assign in_shift = s1;
  assign in_last  = l1;

  always_comb begin
    for (int i = 0; i < ROWS; i++) begin
      logic [7:0] b;
      b = a_rdata[8*i +: 8];
      if (!two_pass)   a_vec[i] = ACT_W'(signed'(b));
      else if (!p1)    a_vec[i] = in_signed ? ACT_W'(signed'(b)) : {1'b0, b};
      else             a_vec[i] = {1'b0, b};
    end
    for (int j = 0; j < COLS; j++) w_vec[j] = signed'(w_rdata[8*j +: 8]);
  end
endmodule
