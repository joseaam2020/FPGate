// End-to-end through the registers: fpgate_accel_core driven only by Avalon-MM slave
// accesses (what the HPS does) with real data from the e1_r7 package in a simulated DDR3.
// Checks: STATUS/DONE, output bytes vs the model, CYCLES, BYTES_RD/BYTES_WR registers,
// an error (odd K) read back through STATUS/ERR_CODE, and a second operation after it.
`timescale 1ns/1ps

module tb_core;
  localparam int A_BASE = 32'h00000;
  localparam int W_BASE = 32'h40000;
  localparam int O_BASE = 32'h80000;
  localparam int DEPTH  = 1 << 15;

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic [6:0]   avs_address;
  logic         avs_write, avs_read, irq;
  logic [31:0]  avs_writedata, avs_readdata;
  logic [3:0]   avs_byteenable;

  logic [31:0]  rd_address, wr_address, wr_byteenable;
  logic         rd_read, rd_waitrequest, rd_readdatavalid, wr_write, wr_waitrequest;
  logic [7:0]   rd_burstcount, wr_burstcount;
  logic [255:0] rd_readdata, wr_writedata;

  // softmax unit's masters: idle here, tied off
  logic [31:0]  rd2_address, wr2_address, wr2_byteenable;
  logic         rd2_read, wr2_write;
  logic [7:0]   rd2_burstcount, wr2_burstcount;
  logic [255:0] wr2_writedata;
  logic         rd2_waitrequest = 0, rd2_readdatavalid = 0, wr2_waitrequest = 0;
  logic [255:0] rd2_readdata = '0;

  fpgate_accel_core #(.A_BEATS(4096), .W_BEATS(1024), .LANES(4), .ADDR_LO(32'h0), .ADDR_HI(32'hFFFF_FFE0)) dut (.*);

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
    avs_address = off[8:2]; avs_writedata = d; avs_byteenable = 4'hF; avs_write = 1; avs_read = 0;
    @(posedge clk); #1;
    avs_write = 0;
  endtask
  task automatic rd(input int off, output logic [31:0] d);
    @(posedge clk); #1;
    avs_address = off[8:2]; avs_read = 1; avs_write = 0; avs_byteenable = 4'hF;
    @(posedge clk); #1;
    avs_read = 0;
    d = avs_readdata;
  endtask
  task automatic expect_eq(input string what, input logic [31:0] got, input logic [31:0] exp);
    checks++;
    if (got !== exp) begin $display("FAIL %s: got 0x%08x, expected 0x%08x", what, got, exp); errors++; end
  endtask

  logic [255:0] img_a [4096];
  logic [255:0] img_w [4 * (8 + 1024)];
  logic [15:0]  y_m [32 * 32];

  task automatic run_layer(input string name, input int spp);
    int R, C, K, ib, ob, fd, rc, stride, subs, nt, planes, bad, exp_rd, exp_wr, left, panel_sub, to;
    logic [31:0] s, v;
    string d;
    d = {"vectors/", name, "/"};
    fd = $fopen({d, "params.txt"}, "r");
    rc = $fscanf(fd, "%d %d %d %d %d", R, C, K, ib, ob);
    $fclose(fd);
    planes = (ib > 8) ? 2 : 1; subs = (R + 15) / 16; nt = C / 16;
    $readmemh({d, "mem_a.hex"}, img_a);
    $readmemh({d, "mem_w.hex"}, img_w);
    $readmemh({d, "y.hex"}, y_m);
    stride = (ob > 8) ? 96 : 64;
    for (int i = 0; i < subs * planes * K / 2; i++) begin mem_r.mem[A_BASE / 32 + i] = img_a[i]; mem_w.mem[A_BASE / 32 + i] = img_a[i]; end
    for (int i = 0; i < nt * (8 + K / 2); i++) begin mem_r.mem[W_BASE / 32 + i] = img_w[i]; mem_w.mem[W_BASE / 32 + i] = img_w[i]; end
    for (int i = 0; i < (R * stride) / 32 + 2; i++) begin mem_r.mem[O_BASE / 32 + i] = {32{8'hCC}}; mem_w.mem[O_BASE / 32 + i] = {32{8'hCC}}; end

    wr(32'h20, A_BASE); wr(32'h24, W_BASE); wr(32'h28, O_BASE); wr(32'h4C, stride);
    wr(32'h14, R); wr(32'h18, C); wr(32'h1C, K);
    wr(32'h10, {16'b0, 8'(spp), 5'b0, (ob > 8), 1'b1, (ib > 8)});
    wr(32'h04, 32'h1);                                  // START
    rd(32'h08, s);
    expect_eq({name, " BUSY after START"}, s & 32'h7, 32'h1);
    to = 0;
    do begin rd(32'h08, s); to++; end while (s[0] && to < 200000);
    expect_eq({name, " STATUS done"}, s, 32'h2);
    expect_eq({name, " irq low (IRQ_EN off)"}, {31'b0, irq}, 32'h0);

    bad = 0;
    for (int r = 0; r < R; r++)
      for (int n = 0; n < C; n++) begin
        logic [255:0] beatv;
        int off, got, exp;
        off = O_BASE + r * stride + n * ((ob > 8) ? 2 : 1);
        beatv = mem_w.mem[off / 32];
        if (ob > 8) begin got = int'(signed'(beatv[8*(off%32) +: 16])); exp = int'(signed'(y_m[r*C+n])); end
        else        begin got = int'(signed'(beatv[8*(off%32) +: 8]));  exp = int'(signed'(y_m[r*C+n][7:0])); end
        if (got !== exp) begin if (bad < 5) $display("FAIL %s [%0d][%0d]: got %0d expected %0d", name, r, n, got, exp); bad++; end
      end
    checks++;
    if (bad != 0) errors++;

    exp_rd = 0; left = subs;
    while (left > 0) begin
      panel_sub = (left < spp) ? left : spp;
      exp_rd += panel_sub * planes * (K / 2) + nt * (8 + K / 2);
      left -= panel_sub;
    end
    exp_rd *= 32;
    exp_wr = R * nt * ((ob > 8) ? 32 : 16);
    rd(32'h40, v); expect_eq({name, " BYTES_RD"}, v, exp_rd);
    rd(32'h44, v); expect_eq({name, " BYTES_WR"}, v, exp_wr);
    rd(32'h38, v);
    checks++;
    if (v < subs * nt * planes * K) begin $display("FAIL %s CYCLES %0d below the pure compute time", name, v); errors++; end
    $display("  %s via CSR: %0d outputs, CYCLES register = %0d", name, R * C, v);
  endtask

  initial begin
    logic [31:0] v;
    void'($urandom(32'h0C0DE));
    avs_address = 0; avs_write = 0; avs_read = 0; avs_writedata = 0; avs_byteenable = 4'hF;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    rd(32'h00, v); expect_eq("VERSION", v, 32'h4650_0002);

    run_layer("b0_q_w", 2);
    run_layer("b0_fc2_w", 1);

    // odd K is rejected and readable through STATUS / ERR_CODE
    wr(32'h14, 16); wr(32'h18, 16); wr(32'h1C, 7); wr(32'h10, 32'h0200);
    wr(32'h04, 32'h1);
    repeat (50) @(posedge clk);
    rd(32'h08, v); expect_eq("STATUS error", v, 32'h4);
    rd(32'h0C, v); expect_eq("ERR_CODE bad parameters", v, 32'h2);
    wr(32'h04, 32'h8);                                  // CLEAR_ERR
    rd(32'h08, v); expect_eq("STATUS after CLEAR_ERR", v, 32'h0);

    run_layer("b0_v_w", 2);

    if (errors == 0) $display("PASS tb_core: %0d checks", checks);
    else             $display("FAIL tb_core: %0d errors in %0d checks", errors, checks);
    $finish;
  end

  initial begin
    #3s;
    $display("FAIL tb_core: global timeout");
    $finish;
  end
endmodule
