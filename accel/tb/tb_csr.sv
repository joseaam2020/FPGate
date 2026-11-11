// Self-checking testbench for fpgate_accel_top (CSR + H0 placeholder engine),
// driven through the Avalon-MM slave the way the HPS lightweight bridge does.
`timescale 1ns/1ps

module tb_csr;
  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic [5:0]  avs_address;
  logic        avs_write, avs_read;
  logic [31:0] avs_writedata, avs_readdata;
  logic [3:0]  avs_byteenable;
  logic        irq;

  fpgate_accel_top dut (.*);

  int errors = 0;
  int checks = 0;

  task automatic wr(input int off, input logic [31:0] d, input logic [3:0] be = 4'hF);
    @(posedge clk); #1;
    avs_address = off[7:2]; avs_writedata = d; avs_byteenable = be; avs_write = 1; avs_read = 0;
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

  task automatic expect_eq(input string what, input int off, input logic [31:0] exp);
    logic [31:0] d;
    rd(off, d);
    checks++;
    if (d !== exp) begin
      $display("FAIL %s (0x%02x): got 0x%08x, expected 0x%08x", what, off, d, exp);
      errors++;
    end
  endtask

  task automatic wait_not_busy(input int max);
    logic [31:0] s;
    int n = 0;
    do begin rd(32'h08, s); n++; end while (s[0] && n < max);
  endtask

  logic [31:0] v, h0, h1, c;

  initial begin
    avs_address = 0; avs_write = 0; avs_read = 0; avs_writedata = 0; avs_byteenable = 4'hF;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    // identification and reset values
    expect_eq("VERSION", 32'h00, 32'h4650_0001);
    expect_eq("STATUS idle", 32'h08, 32'h0);
    expect_eq("ERR_CODE", 32'h0C, 32'h0);
    expect_eq("unmapped", 32'h60, 32'hDEAD_BEEF);

    // read-only registers ignore writes
    wr(32'h00, 32'hFFFF_FFFF);
    wr(32'h08, 32'hFFFF_FFFF);
    expect_eq("VERSION after write", 32'h00, 32'h4650_0001);
    expect_eq("STATUS after write", 32'h08, 32'h0);

    // heartbeat advances
    rd(32'h48, h0); repeat (20) @(posedge clk); rd(32'h48, h1);
    checks++;
    if (h1 <= h0) begin $display("FAIL HEARTBEAT did not advance: %0d -> %0d", h0, h1); errors++; end

    // read/write registers, with byte enables
    wr(32'h50, 32'hA5A5_5A5A);
    expect_eq("SCRATCH", 32'h50, 32'hA5A5_5A5A);
    wr(32'h50, 32'h0000_00FF, 4'b0001);
    expect_eq("SCRATCH byte 0", 32'h50, 32'hA5A5_5AFF);
    wr(32'h50, 32'h1100_0000, 4'b1000);
    expect_eq("SCRATCH byte 3", 32'h50, 32'h11A5_5AFF);
    for (int r = 0; r < 10; r++) begin
      wr(32'h10 + 4*r, 32'h1000_0000 + r);
    end
    // 0x10..0x34 are OP, DIM_M/N/K, ADDR_*, STRIDE_A; DIM_M also drives the dummy engine,
    // so read them back before starting
    for (int r = 0; r < 10; r++) expect_eq($sformatf("reg 0x%02x", 32'h10 + 4*r), 32'h10 + 4*r, 32'h1000_0000 + r);
    wr(32'h4C, 32'h0BAD_F00D);
    expect_eq("STRIDE_OUT", 32'h4C, 32'h0BAD_F00D);

    // START -> BUSY -> DONE, cycle counter
    wr(32'h10, 32'h0);          // OP: no failure
    wr(32'h14, 32'd100);        // DIM_M = 100 busy cycles
    wr(32'h04, 32'h1);          // START
    rd(32'h08, v);
    checks++;
    if (v[0] !== 1'b1) begin $display("FAIL expected BUSY right after START, STATUS=0x%08x", v); errors++; end
    wait_not_busy(1000);
    expect_eq("STATUS done", 32'h08, 32'h2);
    rd(32'h38, c);
    checks++;
    if (c < 100 || c > 130) begin $display("FAIL CYCLES_LO=%0d, expected about 100", c); errors++; end
    expect_eq("CYCLES_HI", 32'h3C, 32'h0);

    // START while busy -> error 1
    wr(32'h14, 32'd400);
    wr(32'h04, 32'h1);
    wr(32'h04, 32'h1);
    expect_eq("STATUS error", 32'h08, 32'h5);      // busy + error
    expect_eq("ERR_CODE start-while-busy", 32'h0C, 32'h1);
    wr(32'h04, 32'h8);                              // CLEAR_ERR
    expect_eq("STATUS after CLEAR_ERR", 32'h08, 32'h1);
    wait_not_busy(2000);
    expect_eq("STATUS done 2", 32'h08, 32'h2);

    // engine error + interrupt
    wr(32'h10, 32'h8000_0000);   // OP[31]: engine fails
    wr(32'h14, 32'd50);
    wr(32'h04, 32'h5);           // START | IRQ_EN
    wait_not_busy(1000);
    expect_eq("STATUS engine error", 32'h08, 32'h4);
    expect_eq("ERR_CODE engine", 32'h0C, 32'h10);
    checks++;
    if (irq !== 1'b1) begin $display("FAIL irq should be high after an error with IRQ_EN"); errors++; end
    wr(32'h04, 32'h8);           // CLEAR_ERR keeping IRQ_EN low (bit 2 = 0)
    expect_eq("STATUS after clear", 32'h08, 32'h0);
    checks++;
    if (irq !== 1'b0) begin $display("FAIL irq should be low"); errors++; end

    // soft reset while busy
    wr(32'h10, 32'h0);
    wr(32'h14, 32'd5000);
    wr(32'h04, 32'h1);
    wr(32'h04, 32'h2);           // RESET_SOFT
    expect_eq("STATUS after RESET_SOFT", 32'h08, 32'h0);
    repeat (50) @(posedge clk);
    expect_eq("still idle (engine reset too)", 32'h08, 32'h0);

    if (errors == 0) $display("PASS tb_csr: %0d checks", checks);
    else             $display("FAIL tb_csr: %0d errors in %0d checks", errors, checks);
    $finish;
  end

  initial begin
    #5ms;
    $display("FAIL tb_csr: global timeout");
    $finish;
  end
endmodule
