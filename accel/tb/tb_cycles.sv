// Cycle count of one full operation through the registers, with the same dimensions
// and sub-tiles per panel that accel_test used on the board. Timing in the engine does
// not depend on data values, so the memory holds zeros. Parameters are overridden from
// the command line (vsim -gM=... ), see tools/sim_cycles.ps1.
`timescale 1ns/1ps

module tb_cycles #(
  parameter int M        = 32,
  parameter int N        = 32,
  parameter int K        = 768,
  parameter int IB       = 8,
  parameter int OB       = 16,
  parameter int SPP      = 2,
  parameter int WAIT_PCT = 0,
  parameter int GAP_PCT  = 0,
  parameter int LAT      = 4,
  parameter int RAW      = 0      // 1: RAW32 output (attention scores)
);
  localparam int PLANES  = (IB > 8) ? 2 : 1;
  localparam int SUBS    = (M + 15) / 16;
  localparam int NT      = N / 16;
  localparam int A_BEATS = SUBS * PLANES * K / 2;
  localparam int W_BEATS = NT * (8 + K / 2);
  localparam int STRIDE  = N * (RAW ? 4 : ((OB > 8) ? 2 : 1));
  localparam int O_BEATS = (M * STRIDE) / 32 + 2;
  localparam int A_BASE  = 0;
  localparam int W_BASE  = ((A_BEATS * 32 + 4095) / 4096) * 4096;
  localparam int O_BASE  = W_BASE + ((W_BEATS * 32 + 4095) / 4096) * 4096;
  localparam int DEPTH   = O_BASE / 32 + O_BEATS + 64;

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

  fpgate_accel_core #(.A_BEATS(8192), .W_BEATS(1024), .LANES(4), .SM_LANES(4)) dut (.*);
  cyc_mon #(.SM(0)) mon();

  avalon_mem_model #(.DEPTH(DEPTH), .WAIT_PCT(WAIT_PCT), .GAP_PCT(GAP_PCT), .LAT(LAT)) mem_r (
    .clk, .rst_n, .address(rd_address), .read(rd_read), .write(1'b0), .writedata('0),
    .byteenable('0), .burstcount(rd_burstcount), .waitrequest(rd_waitrequest),
    .readdata(rd_readdata), .readdatavalid(rd_readdatavalid), .hang_reads(1'b0)
  );
  avalon_mem_model #(.DEPTH(DEPTH), .WAIT_PCT(WAIT_PCT), .GAP_PCT(GAP_PCT), .LAT(LAT)) mem_w (
    .clk, .rst_n, .address(wr_address), .read(1'b0), .write(wr_write), .writedata(wr_writedata),
    .byteenable(wr_byteenable), .burstcount(wr_burstcount), .waitrequest(wr_waitrequest),
    .readdata(), .readdatavalid(), .hang_reads(1'b0)
  );

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

  initial begin
    logic [31:0] s, cyc, brd, bwr;
    int to;
    avs_address = 0; avs_write = 0; avs_read = 0; avs_writedata = 0; avs_byteenable = 4'hF;
    for (int i = 0; i < DEPTH; i++) begin mem_r.mem[i] = '0; mem_w.mem[i] = '0; end
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    wr(32'h20, A_BASE); wr(32'h24, W_BASE); wr(32'h28, O_BASE); wr(32'h4C, STRIDE);
    wr(32'h14, M); wr(32'h18, N); wr(32'h1C, K);
    wr(32'h10, {16'b0, 8'(SPP), 4'b0, 1'(RAW), (OB > 8), 1'b1, (IB > 8)});
    mon.start();
    wr(32'h04, 32'h1);                                  // START
    to = 0;
    do begin rd(32'h08, s); to++; end while (s[0] && to < 2000000);
    if (RAW) mon.report("S"); else mon.report("PV");
    rd(32'h38, cyc); rd(32'h40, brd); rd(32'h44, bwr);
    $display("RESULT M=%0d N=%0d K=%0d in=%0d out=%0d spp=%0d wait=%0d gap=%0d lat=%0d status=0x%0h cycles=%0d rd=%0d wr=%0d",
             M, N, K, IB, OB, SPP, WAIT_PCT, GAP_PCT, LAT, s, cyc, brd, bwr);
    $display("PASS tb_cycles");
    $finish;
  end

  initial begin
    #(2000000000);   // 2 s of simulated time (300 M cycles) is far above any case here
    $display("FAIL tb_cycles: global timeout");
    $finish;
  end
endmodule
