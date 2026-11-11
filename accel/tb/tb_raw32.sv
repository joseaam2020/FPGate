// RAW32 output mode: the engine delivers the INT32 accumulators of Q.K^T unchanged (64 bytes per row
// of a 16-column tile). Real attention data of the e1_r7 model (block 0 head 0 and block 11 head 5,
// written by `fpgate-accel --dump-attn`, packed by tools/pack_attn_s.py) go through fpgate_accel_core
// by registers only; the scores in the simulated DDR3 must equal the reference bit by bit.
// The run directory is accel/ (sim.ps1 / sim_cycles.ps1 start there).
`timescale 1ns/1ps

module tb_raw32;
  localparam int A_BASE = 32'h00000;
  localparam int W_BASE = 32'h08000;
  localparam int O_BASE = 32'h10000;
  localparam int DEPTH  = 1 << 12;

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic [5:0]   avs_address;
  logic         avs_write, avs_read, irq;
  logic [31:0]  avs_writedata, avs_readdata;
  logic [3:0]   avs_byteenable;

  logic [31:0]  rd_address, wr_address, wr_byteenable;
  logic         rd_read, rd_waitrequest, rd_readdatavalid, wr_write, wr_waitrequest;
  logic [7:0]   rd_burstcount, wr_burstcount;
  logic [255:0] rd_readdata, wr_writedata;

  fpgate_accel_core #(.A_BEATS(4096), .W_BEATS(1024), .LANES(4)) dut (.*);

  avalon_mem_model #(.DEPTH(DEPTH)) mem_r (
    .clk, .rst_n, .address(rd_address), .read(rd_read), .write(1'b0), .writedata('0),
    .byteenable('0), .burstcount(rd_burstcount), .waitrequest(rd_waitrequest),
    .readdata(rd_readdata), .readdatavalid(rd_readdatavalid), .hang_reads(1'b0)
  );
  avalon_mem_model #(.DEPTH(DEPTH)) mem_w (
    .clk, .rst_n, .address(wr_address), .read(1'b0), .write(wr_write), .writedata(wr_writedata),
    .byteenable(wr_byteenable), .burstcount(wr_burstcount), .waitrequest(wr_waitrequest),
    .readdata(), .readdatavalid(), .hang_reads(1'b0)
  );

  int errors = 0, checks = 0;

  task automatic wr(input int off, input logic [31:0] d);
    @(posedge clk); #1;
    avs_address = off[7:2]; avs_writedata = d; avs_byteenable = 4'hF; avs_write = 1; avs_read = 0;
    @(posedge clk); #1;
    avs_write = 0;
  endtask
  task automatic rd(input int off, output logic [31:0] d);
    @(posedge clk); #1;
    avs_address = off[7:2]; avs_read = 1; avs_write = 0; avs_byteenable = 4'hF;
    @(posedge clk); #1;
    avs_read = 0;
    d = avs_readdata;
  endtask
  task automatic expect_eq(input string what, input logic [31:0] got, input logic [31:0] exp);
    checks++;
    if (got !== exp) begin $display("FAIL %s: got 0x%08x, expected 0x%08x", what, got, exp); errors++; end
  endtask

  logic [255:0] img_a [512];
  logic [255:0] img_w [512];
  logic [31:0]  s_ref [32 * 48];

  // one operation over the first M rows and N keys of the packed problem (the arrays hold 32 x 48)
  task automatic run_case(input string name, input int M, input int N, input int spp);
    logic [31:0] s, v;
    int to, bad, stride, K;
    stride = N * 4;
    K = 64;
    wr(32'h04, 32'h2);                                  // RESET_SOFT
    for (int i = 0; i < DEPTH; i++) begin mem_r.mem[i] = '0; mem_w.mem[i] = '0; end
    for (int i = 0; i < 64; i++) begin mem_r.mem[A_BASE / 32 + i] = img_a[i]; end
    for (int i = 0; i < 3 * 40; i++) begin mem_r.mem[W_BASE / 32 + i] = img_w[i]; end
    for (int i = 0; i < (M * stride) / 32 + 2; i++) begin mem_r.mem[O_BASE / 32 + i] = {32{8'hCC}}; mem_w.mem[O_BASE / 32 + i] = {32{8'hCC}}; end

    wr(32'h20, A_BASE); wr(32'h24, W_BASE); wr(32'h28, O_BASE); wr(32'h4C, stride);
    wr(32'h14, M); wr(32'h18, N); wr(32'h1C, K);
    wr(32'h10, {16'b0, 8'(spp), 4'b0, 1'b1, 1'b0, 1'b1, 1'b0});   // raw32, signed 8-bit input, 8-bit planes
    wr(32'h04, 32'h1);                                  // START
    to = 0;
    do begin rd(32'h08, s); to++; end while (s[0] && to < 200000);
    expect_eq({name, " STATUS done"}, s, 32'h2);

    bad = 0;
    for (int r = 0; r < M; r++)
      for (int n = 0; n < N; n++) begin
        int off;
        logic [31:0] got, exp;
        off = O_BASE + r * stride + n * 4;
        got = mem_w.mem[off / 32][8 * (off % 32) +: 32];
        exp = s_ref[r * 48 + n];
        if (got !== exp) begin if (bad < 5) $display("FAIL %s [%0d][%0d]: got %0d expected %0d", name, r, n, $signed(got), $signed(exp)); bad++; end
      end
    checks++;
    if (bad != 0) begin errors++; $display("FAIL %s: %0d of %0d scores differ", name, bad, M * N); end
    else $display("  %s: %0d x %0d scores identical to the reference", name, M, N);
    rd(32'h44, v); expect_eq({name, " BYTES_WR"}, v, M * (N / 16) * 64);
    rd(32'h38, v);
    $display("    CYCLES = %0d", v);
    // rows past the last valid row, and bytes after the matrix, are untouched
    expect_eq({name, " byte after the matrix"}, mem_w.mem[(O_BASE + M * stride) / 32][7:0], 32'hCC);
  endtask

  task automatic run_dir(input string dir);
    int fd, rc, R, Kc, Hd;
    $readmemh({dir, "/raw32_a.hex"}, img_a);
    $readmemh({dir, "/raw32_w.hex"}, img_w);
    $readmemh({dir, "/raw32_s.hex"}, s_ref);
    fd = $fopen({dir, "/raw32_params.txt"}, "r");
    rc = $fscanf(fd, "%d %d %d", R, Kc, Hd);
    $fclose(fd);
    $display("%s: %0d rows x %0d keys, K = %0d", dir, R, Kc, Hd);
    run_case({dir, " 32x48 spp2"}, 32, 48, 2);
    run_case({dir, " 20x32 spp1"}, 20, 32, 1);
    run_case({dir, " 16x16 spp1"}, 16, 16, 1);
  endtask

  initial begin
    logic [31:0] v;
    void'($urandom(32'h0C0DE));
    avs_address = 0; avs_write = 0; avs_read = 0; avs_writedata = 0; avs_byteenable = 4'hF;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    rd(32'h00, v); expect_eq("VERSION", v, 32'h4650_0001);
    run_dir("vectors/attn_b0_h0");
    run_dir("vectors/attn_b11_h5");

    if (errors == 0) $display("PASS tb_raw32: %0d checks", checks);
    else             $display("FAIL tb_raw32: %0d errors in %0d checks", errors, checks);
    $finish;
  end

  initial begin
    #(200000000);
    $display("FAIL tb_raw32: global timeout");
    $finish;
  end
endmodule
