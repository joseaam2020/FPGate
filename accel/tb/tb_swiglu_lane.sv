// Self-checking testbench for swiglu_lane.
//  1) Real data: blocks 0 and 11 of image 0 (gi, ui -> expected swiglu_i), bit-exact.
//  2) Synthetic: random table, random inputs, shifts 30..42, saturation; 64-bit reference.
`timescale 1ns/1ps

module tb_swiglu_lane;
  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic               tbl_we;
  logic        [7:0]  tbl_addr;
  logic signed [15:0] tbl_data;
  logic               in_valid;
  logic signed [7:0]  gi, ui;
  logic        [30:0] m0;
  logic        [5:0]  shift;
  logic               out_valid;
  logic signed [15:0] h;

  swiglu_lane dut (.*);

  int errors = 0;
  int checked = 0;
  logic signed [15:0] expq [$];

  always @(posedge clk) begin
    if (out_valid) begin
      logic signed [15:0] e;
      e = expq.pop_front();
      checked++;
      if (h !== e) begin
        if (errors < 10) $display("FAIL: got %0d, expected %0d", h, e);
        errors++;
      end
    end
  end

  logic signed [15:0] tbl_ref [256];

  task automatic load_table();
    for (int i = 0; i < 255; i++) begin
      @(posedge clk); #1;
      tbl_we = 1; tbl_addr = i[7:0]; tbl_data = tbl_ref[i];
    end
    @(posedge clk); #1;
    tbl_we = 0;
  endtask

  function automatic logic signed [15:0] ref_h(input int g, input int u, input int unsigned mm, input int s);
    longint p, r;
    p = longint'(tbl_ref[g + 127]) * longint'(u);
    r = (p * longint'(mm) + (64'sd1 << (s - 1))) >>> s;
    if (r > 32767) r = 32767;
    if (r < -32767) r = -32767;
    return 16'(r);
  endfunction

  task automatic drive(input int g, input int u, input int unsigned mm, input int s);
    @(posedge clk); #1;
    in_valid = 1; gi = g[7:0]; ui = u[7:0]; m0 = mm[30:0]; shift = s[5:0];
    expq.push_back(ref_h(g, u, mm, s));
  endtask

  task automatic idle(input int n);
    repeat (n) begin @(posedge clk); #1; in_valid = 0; end
  endtask

  logic [7:0]  g_m [1024];
  logic [7:0]  u_m [1024];
  logic [15:0] h_m [1024];
  logic [15:0] t_m [256];

  task automatic run_block(input string name);
    int R, C, fd, rc;
    int unsigned mm, ss;
    string d;
    d = {"vectors/", name, "/"};
    fd = $fopen({d, "params.txt"}, "r");
    if (fd == 0) begin $display("FAIL: cannot open %sparams.txt (run tools/dump_swiglu.py)", d); errors++; return; end
    rc = $fscanf(fd, "%d %d %d %d", R, C, mm, ss);
    $fclose(fd);
    $readmemh({d, "gi.hex"}, g_m);
    $readmemh({d, "ui.hex"}, u_m);
    $readmemh({d, "h.hex"}, h_m);
    $readmemh({d, "silu.hex"}, t_m);
    for (int i = 0; i < 255; i++) tbl_ref[i] = t_m[i];
    load_table();
    for (int i = 0; i < R*C; i++) begin
      // expected comes from the package, not from ref_h
      @(posedge clk); #1;
      in_valid = 1; gi = g_m[i]; ui = u_m[i]; m0 = mm[30:0]; shift = ss[5:0];
      expq.push_back(h_m[i]);
    end
    idle(14);
    $display("  %s: %0d values", name, R*C);
  endtask

  initial begin
    void'($urandom(32'h5111C0DE));
    tbl_we = 0; tbl_addr = 0; tbl_data = 0;
    in_valid = 0; gi = 0; ui = 0; m0 = 0; shift = 1;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    run_block("b0_swiglu");
    run_block("b11_swiglu");

    // synthetic
    for (int i = 0; i < 255; i++) tbl_ref[i] = 16'($urandom_range(65535, 0));
    load_table();
    for (int n = 0; n < 20000; n++) begin
      int g, u, s;
      int unsigned mm;
      g  = $urandom_range(254, 0) - 127;
      u  = $urandom_range(254, 0) - 127;
      mm = 32'h4000_0000 + $urandom_range(32'h3fff_ffff, 0);
      s  = $urandom_range(42, 30);
      if ($urandom_range(9, 0) == 0) s = $urandom_range(62, 20);   // also exercise saturation
      drive(g, u, mm, s);
      if ($urandom_range(7, 0) == 0) idle($urandom_range(3, 1));
    end
    drive(-127, -127, 32'h7fffffff, 30);
    drive(127, 127, 32'h7fffffff, 30);
    drive(0, 0, 32'h4000_0000, 30);
    idle(14);

    if (expq.size() != 0) begin $display("FAIL: %0d results never came out", expq.size()); errors++; end
    if (errors == 0) $display("PASS tb_swiglu_lane: %0d values", checked);
    else             $display("FAIL tb_swiglu_lane: %0d errors in %0d values", errors, checked);
    $finish;
  end

  initial begin
    #20ms;
    $display("FAIL tb_swiglu_lane: global timeout");
    $finish;
  end
endmodule
