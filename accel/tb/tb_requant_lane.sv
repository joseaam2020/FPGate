// Self-checking testbench for requant_lane.
//  1) Real data: slices dumped from the e1_r7 package by tools/dump_vectors.py
//     (acc + bias -> expected y from golden.requantize), bit-exact.
//  2) Synthetic: random and extreme values against a 64-bit reference, so that
//     saturation and the whole shift range (1..62) are exercised.
`timescale 1ns/1ps

module tb_requant_lane;
  localparam int MAXN = 4096;

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic               in_valid;
  logic signed [31:0] acc, bias;
  logic        [30:0] m0;
  logic        [5:0]  shift;
  logic               out16, out_valid;
  logic signed [15:0] y;

  requant_lane dut (.*);

  int errors = 0;
  int checked = 0;

  // expected results queue, filled when a value is driven
  logic signed [15:0] expq [$];

  always @(posedge clk) begin
    if (out_valid) begin
      logic signed [15:0] e;
      e = expq.pop_front();
      checked++;
      if (y !== e) begin
        if (errors < 10) $display("FAIL: got %0d, expected %0d", y, e);
        errors++;
      end
    end
  end

  // 64-bit reference (product < 2^62 fits a signed longint)
  function automatic logic signed [15:0] ref_rq(input int a, input int b, input int unsigned mm,
                                                 input int s, input bit wide);
    longint p, qm, r;
    p  = (longint'(a) + longint'(b)) * longint'(mm);
    r  = (p + (64'sd1 << (s - 1))) >>> s;
    qm = wide ? 32767 : 127;
    if (r > qm) r = qm;
    if (r < -qm) r = -qm;
    return 16'(r);
  endfunction

  task automatic drive(input int a, input int b, input int unsigned mm, input int s, input bit wide,
                       input logic signed [15:0] exp_y);
    @(posedge clk); #1;
    in_valid = 1; acc = a; bias = b; m0 = mm[30:0]; shift = s[5:0]; out16 = wide;
    expq.push_back(exp_y);
  endtask

  task automatic idle(input int n);
    repeat (n) begin @(posedge clk); #1; in_valid = 0; end
  endtask

  // ---- real data -----------------------------------------------------------
  logic [31:0] acc_m [MAXN];
  logic [31:0] bias_m [128];
  logic [31:0] m0_m [128];
  logic [7:0]  sh_m [128];
  logic [15:0] y_m [MAXN];

  task automatic run_layer(input string name);
    int R, C, K, ib, ob, fd, rc;
    string d;
    d = {"vectors/", name, "/"};
    fd = $fopen({d, "params.txt"}, "r");
    if (fd == 0) begin $display("FAIL: cannot open %sparams.txt (run tools/dump_vectors.py)", d); errors++; return; end
    rc = $fscanf(fd, "%d %d %d %d %d", R, C, K, ib, ob);
    $fclose(fd);
    $readmemh({d, "acc.hex"}, acc_m);
    $readmemh({d, "bias.hex"}, bias_m);
    $readmemh({d, "m0.hex"}, m0_m);
    $readmemh({d, "shift.hex"}, sh_m);
    $readmemh({d, "y.hex"}, y_m);
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        drive(int'(acc_m[r*C+c]), int'(bias_m[c]), m0_m[c], int'(sh_m[c]), ob > 8,
              ob > 8 ? y_m[r*C+c] : 16'(signed'(y_m[r*C+c][7:0])));
    idle(12);
    $display("  layer %s: %0d values", name, R*C);
  endtask

  initial begin
    void'($urandom(32'h00E91106));
    in_valid = 0; acc = 0; bias = 0; m0 = 0; shift = 1; out16 = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    run_layer("b0_q_w");
    run_layer("b0_v_w");
    run_layer("b0_fc1g_w");
    run_layer("b0_fc2_w");
    run_layer("b1_fc2_w");

    // synthetic: random values, every shift, both widths, with bubbles
    for (int n = 0; n < 20000; n++) begin
      int a, b, s;
      int unsigned mm;
      bit w;
      a  = $urandom();
      b  = $urandom_range(1000, 0) - 500;
      if ($urandom_range(3, 0) == 0) a = a >>> $urandom_range(24, 0);   // mix magnitudes
      mm = 32'h4000_0000 + $urandom_range(32'h3fff_ffff, 0);             // [2^30, 2^31)
      s  = $urandom_range(62, 1);
      w  = $urandom_range(1, 0);
      // keep acc + bias inside INT32 like the real accumulator
      if (longint'(a) + longint'(b) > 64'sh7fffffff || longint'(a) + longint'(b) < -64'sh80000000) b = 0;
      drive(a, b, mm, s, w, ref_rq(a, b, mm, s, w));
      if ($urandom_range(7, 0) == 0) idle($urandom_range(3, 1));
    end

    // extremes: saturation both ways, ties (round half up), zero
    drive(32'h7fffffff, 0, 32'h7fffffff, 31, 1, ref_rq(32'h7fffffff, 0, 32'h7fffffff, 31, 1));
    drive(-32'sd2147483647 - 1, 0, 32'h7fffffff, 31, 0, ref_rq(-32'sd2147483647 - 1, 0, 32'h7fffffff, 31, 0));
    drive(0, 0, 32'h4000_0000, 30, 1, ref_rq(0, 0, 32'h4000_0000, 30, 1));
    drive(1, 0, 32'h4000_0000, 31, 1, ref_rq(1, 0, 32'h4000_0000, 31, 1));    // +0.5 -> rounds up to 1
    drive(-1, 0, 32'h4000_0000, 31, 1, ref_rq(-1, 0, 32'h4000_0000, 31, 1));  // -0.5 -> rounds up to 0
    idle(12);

    if (expq.size() != 0) begin $display("FAIL: %0d results never came out", expq.size()); errors++; end
    if (errors == 0) $display("PASS tb_requant_lane: %0d values", checked);
    else             $display("FAIL tb_requant_lane: %0d errors in %0d values", errors, checked);
    $finish;
  end

  initial begin
    #20ms;
    $display("FAIL tb_requant_lane: global timeout");
    $finish;
  end
endmodule
