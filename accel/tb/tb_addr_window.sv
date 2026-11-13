// Address window guard: bursts / beats outside [0x3000_0000, 0x4000_0000) never reach the bus.
// Core level (through the registers): operand read at 0x2FFF_FFE0 ends with STATUS.ERROR and
// ERR_CODE 7, with no read or write on any master. Unit level: dma_rd and dma_wr pulse err_win and
// stay off the bus for a bad address, and still work for a good one.
`timescale 1ns/1ps

module tb_addr_window;
  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  // ---- core ----
  logic [6:0]   avs_address;
  logic         avs_write, avs_read, irq;
  logic [31:0]  avs_writedata, avs_readdata;
  logic [3:0]   avs_byteenable;
  logic [31:0]  rd_address, wr_address, wr_byteenable, rd2_address, wr2_address, wr2_byteenable;
  logic         rd_read, wr_write, rd2_read, wr2_write;
  logic [7:0]   rd_burstcount, wr_burstcount, rd2_burstcount, wr2_burstcount;
  logic [255:0] wr_writedata, wr2_writedata;
  logic         rd_waitrequest = 0, rd_readdatavalid = 0, wr_waitrequest = 0;
  logic         rd2_waitrequest = 0, rd2_readdatavalid = 0, wr2_waitrequest = 0;
  logic [255:0] rd_readdata = '0, rd2_readdata = '0;

  fpgate_accel_core #(.A_BEATS(4096), .W_BEATS(1024), .LANES(4), .SM_LANES(4)) dut (.*);

  int bus_ops = 0;
  always @(posedge clk) if (rd_read || wr_write || rd2_read || wr2_write) bus_ops++;

  int errors = 0, checks = 0;
  task automatic expect_eq(input string what, input logic [31:0] got, input logic [31:0] exp);
    checks++;
    if (got !== exp) begin $display("FAIL %s: got 0x%08x, expected 0x%08x", what, got, exp); errors++; end
  endtask
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

  // ---- units ----
  logic         flush = 0, rcmd_valid = 0, rcmd_ready, rbusy, rdone, rerr, rwin, rbv;
  logic [31:0]  rcmd_addr = 0;
  logic [15:0]  rcmd_beats = 0, rbi;
  logic [255:0] rbd;
  logic [31:0]  ru_addr;
  logic         ru_read;
  logic [7:0]   ru_bc;
  dma_rd #(.TIMEOUT(200)) u_rd (
    .clk, .rst_n, .flush, .cmd_valid(rcmd_valid), .cmd_ready(rcmd_ready), .cmd_addr(rcmd_addr),
    .cmd_beats(rcmd_beats), .busy(rbusy), .done(rdone), .err(rerr), .err_win(rwin),
    .beat_valid(rbv), .beat_data(rbd), .beat_idx(rbi),
    .m_address(ru_addr), .m_read(ru_read), .m_burstcount(ru_bc), .m_waitrequest(1'b0),
    .m_readdata(256'd0), .m_readdatavalid(1'b0));
  int ru_reads = 0, rwin_n = 0;
  always @(posedge clk) begin if (ru_read) ru_reads++; if (rwin) rwin_n++; end

  logic         wvalid = 0, wready, wbusy, wbeat, werr, wwin;
  logic [31:0]  waddr = 0, wu_addr, wu_be;
  logic [255:0] wu_data;
  logic         wu_write;
  logic [7:0]   wu_bc;
  dma_wr #(.TIMEOUT(200)) u_wr (
    .clk, .rst_n, .flush, .in_valid(wvalid), .in_ready(wready), .in_addr(waddr),
    .in_data(256'hAB), .in_be(32'hFFFF_FFFF), .busy(wbusy), .wr_beat(wbeat), .err(werr), .err_win(wwin),
    .m_address(wu_addr), .m_write(wu_write), .m_writedata(wu_data), .m_byteenable(wu_be),
    .m_burstcount(wu_bc), .m_waitrequest(1'b0));
  int wu_writes = 0, wwin_n = 0;
  always @(posedge clk) begin if (wu_write) wu_writes++; if (wwin) wwin_n++; end

  task automatic rd_cmd(input logic [31:0] a, input int beats);
    @(posedge clk); #1; rcmd_valid = 1; rcmd_addr = a; rcmd_beats = beats[15:0];
    @(posedge clk); #1; rcmd_valid = 0;
    repeat (20) @(posedge clk);
  endtask
  task automatic wr_beat(input logic [31:0] a);
    @(posedge clk); #1; wvalid = 1; waddr = a;
    @(posedge clk); #1; wvalid = 0;
    repeat (5) @(posedge clk);
  endtask

  initial begin
    logic [31:0] v;
    avs_address = 0; avs_write = 0; avs_read = 0; avs_writedata = 0; avs_byteenable = 4'hF;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    // core: A operand just below the window
    wr(32'h14, 16); wr(32'h18, 16); wr(32'h1C, 16); wr(32'h10, 32'h0100);
    wr(32'h20, 32'h2FFF_FFE0); wr(32'h24, 32'h3000_0000); wr(32'h28, 32'h3010_0000); wr(32'h4C, 32);
    wr(32'h04, 32'h1);
    repeat (200) @(posedge clk);
    rd(32'h08, v); expect_eq("core STATUS error", v, 32'h4);
    rd(32'h0C, v); expect_eq("core ERR_CODE window", v, 32'h7);
    expect_eq("core bus ops", bus_ops, 0);
    wr(32'h04, 32'h8);
    rd(32'h08, v); expect_eq("core STATUS after CLEAR_ERR", v, 32'h0);

    // dma_rd: below the window, running past the end, and a good burst
    rd_cmd(32'h2FFF_FFE0, 4);
    expect_eq("rd below: err_win", rwin_n, 1);
    expect_eq("rd below: no read", ru_reads, 0);
    rd_cmd(32'h4000_0000, 1);                       // first address above the window
    expect_eq("rd above: err_win", rwin_n, 2);
    expect_eq("rd above: no read", ru_reads, 0);
    rd_cmd(32'h3FFF_FFE0, 1);                       // last beat inside the window
    expect_eq("rd last beat: no err_win", rwin_n, 2);
    expect_eq("rd last beat: read issued", ru_reads, 1);

    // dma_wr
    wr_beat(32'h2FFF_FFE0);
    expect_eq("wr below: err_win", wwin_n, 1);
    expect_eq("wr below: no write", wu_writes, 0);
    wr_beat(32'h4000_0000);
    expect_eq("wr above: err_win", wwin_n, 2);
    expect_eq("wr above: no write", wu_writes, 0);
    wr_beat(32'h3FFF_FFE0);
    expect_eq("wr last beat: no err_win", wwin_n, 2);
    expect_eq("wr last beat: write issued", wu_writes > 0, 1);

    if (errors == 0) $display("PASS tb_addr_window: %0d checks", checks);
    else             $display("FAIL tb_addr_window: %0d errors in %0d checks", errors, checks);
    $finish;
  end

  initial begin
    #1ms;
    $display("FAIL tb_addr_window: global timeout");
    $finish;
  end
endmodule
