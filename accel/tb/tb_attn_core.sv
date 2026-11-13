// One attention head through the registers of fpgate_accel_core, step by step like the HPS will do it:
//   1. S = Q.K^T          engine, RAW32 output (INT32 scores, 4160-byte rows)
//   2. P = softmax(S)     softmax unit (registers at 0x100), P in the activation format of the engine
//   3. ctx = P.V          engine, 16-bit unsigned input, INT8 output with the block's av_requant constants
// Real data of block 0 head 0 and block 11 head 5 (tools/pack_attn_full.py). The DDR3 is modelled by four
// memory models (one per master); the testbench copies a result into the model that reads it next, which is
// what a shared DDR3 does. The final context must equal the reference stage bit by bit.
`timescale 1ns/1ps

module tb_attn_core;
  localparam int DEPTH = 1 << 15;
  localparam int A_Q   = 32'h000000;
  localparam int W_K   = 32'h001000;
  localparam int S_OUT = 32'h020000;
  localparam int P_OUT = 32'h050000;
  localparam int W_V   = 32'h070000;
  localparam int CTX   = 32'h090000;
  localparam int KP    = 1026;

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic [6:0]   avs_address;
  logic         avs_write, avs_read, irq;
  logic [31:0]  avs_writedata, avs_readdata;
  logic [3:0]   avs_byteenable;

  logic [31:0]  rd_address, wr_address, wr_byteenable, rd2_address, wr2_address, wr2_byteenable;
  logic         rd_read, rd_waitrequest, rd_readdatavalid, wr_write, wr_waitrequest;
  logic         rd2_read, rd2_waitrequest, rd2_readdatavalid, wr2_write, wr2_waitrequest;
  logic [7:0]   rd_burstcount, wr_burstcount, rd2_burstcount, wr2_burstcount;
  logic [255:0] rd_readdata, wr_writedata, rd2_readdata, wr2_writedata;

  fpgate_accel_core #(.A_BEATS(8192), .W_BEATS(1024), .LANES(4), .SM_LANES(4), .ADDR_LO(32'h0), .ADDR_HI(32'hFFFF_FFE0)) dut (.*);
  cyc_mon mon();

  avalon_mem_model #(.DEPTH(DEPTH)) m_er (
    .clk, .rst_n, .address(rd_address), .read(rd_read), .write(1'b0), .writedata('0),
    .byteenable('0), .burstcount(rd_burstcount), .waitrequest(rd_waitrequest),
    .readdata(rd_readdata), .readdatavalid(rd_readdatavalid), .hang_reads(1'b0));
  avalon_mem_model #(.DEPTH(DEPTH)) m_ew (
    .clk, .rst_n, .address(wr_address), .read(1'b0), .write(wr_write), .writedata(wr_writedata),
    .byteenable(wr_byteenable), .burstcount(wr_burstcount), .waitrequest(wr_waitrequest),
    .readdata(), .readdatavalid(), .hang_reads(1'b0));
  avalon_mem_model #(.DEPTH(DEPTH)) m_sr (
    .clk, .rst_n, .address(rd2_address), .read(rd2_read), .write(1'b0), .writedata('0),
    .byteenable('0), .burstcount(rd2_burstcount), .waitrequest(rd2_waitrequest),
    .readdata(rd2_readdata), .readdatavalid(rd2_readdatavalid), .hang_reads(1'b0));
  avalon_mem_model #(.DEPTH(DEPTH)) m_sw (
    .clk, .rst_n, .address(wr2_address), .read(1'b0), .write(wr2_write), .writedata(wr2_writedata),
    .byteenable(wr2_byteenable), .burstcount(wr2_burstcount), .waitrequest(wr2_waitrequest),
    .readdata(), .readdatavalid(), .hang_reads(1'b0));

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
  // wait until STATUS (at status_off) shows DONE or ERROR
  task automatic wait_done(input string what, input int status_off, input int limit);
    logic [31:0] s;
    int to;
    to = 0;
    do begin rd(status_off, s); to++; end while (s[0] && to < limit);
    expect_eq({what, " STATUS done"}, s, 32'h2);
  endtask

  logic [255:0] img_a [64];
  logic [255:0] img_wk [65 * 40];
  logic [255:0] img_wv [4 * 521];
  logic [7:0]   ctx_ref [32 * 64];

  task automatic run_dir(input string dir);
    int fd, rc, R, bad;
    longint m0v, shv;
    logic [31:0] v;
    $readmemh({dir, "/full_a.hex"}, img_a);
    $readmemh({dir, "/full_wk.hex"}, img_wk);
    $readmemh({dir, "/full_wv.hex"}, img_wv);
    $readmemh({dir, "/full_ctx.hex"}, ctx_ref);
    fd = $fopen({dir, "/full_params.txt"}, "r");
    rc = $fscanf(fd, "%d %d %d", R, m0v, shv);
    $fclose(fd);
    $display("%s: %0d query rows, 1025 keys", dir, R);

    wr(32'h004, 32'h2); wr(32'h104, 32'h2);                 // RESET_SOFT of both units
    for (int i = 0; i < DEPTH; i++) begin m_er.mem[i] = '0; m_ew.mem[i] = '0; m_sr.mem[i] = '0; m_sw.mem[i] = '0; end
    for (int i = 0; i < 64; i++)      m_er.mem[A_Q / 32 + i] = img_a[i];
    for (int i = 0; i < 65 * 40; i++) m_er.mem[W_K / 32 + i] = img_wk[i];
    for (int i = 0; i < 4 * 521; i++) m_er.mem[W_V / 32 + i] = img_wv[i];

    // 1. scores
    wr(32'h020, A_Q); wr(32'h024, W_K); wr(32'h028, S_OUT); wr(32'h04C, 1040 * 4);
    wr(32'h014, R); wr(32'h018, 1040); wr(32'h01C, 64);
    wr(32'h010, {16'b0, 8'd2, 4'b0, 4'b1010});               // spp 2, raw32, signed, 8-bit planes
    mon.start();
    wr(32'h004, 32'h1);
    wait_done("scores", 32'h008, 400000);
    mon.report($sformatf("S_%0drows", R));
    rd(32'h038, v); $display("    scores: %0d cycles", v);
    for (int i = 0; i < R * 130; i++) m_sr.mem[S_OUT / 32 + i] = m_ew.mem[S_OUT / 32 + i];

    // 2. softmax
    wr(32'h110, R); wr(32'h114, 1025); wr(32'h118, KP);
    wr(32'h11C, S_OUT); wr(32'h120, 1040 * 4); wr(32'h124, P_OUT);
    wr(32'h128, 32'(m0v)); wr(32'h12C, 32'(shv));
    mon.start();
    wr(32'h104, 32'h1);
    wait_done("softmax", 32'h108, 400000);
    mon.report($sformatf("softmax_%0drows", R));
    rd(32'h130, v); $display("    softmax: %0d cycles", v);
    for (int i = 0; i < ((R + 15) / 16) * KP; i++) m_er.mem[P_OUT / 32 + i] = m_sw.mem[P_OUT / 32 + i];

    // 3. context
    wr(32'h020, P_OUT); wr(32'h024, W_V); wr(32'h028, CTX); wr(32'h04C, 64);
    wr(32'h014, R); wr(32'h018, 64); wr(32'h01C, KP);
    wr(32'h010, {16'b0, 8'd2, 4'b0, 4'b0001});               // spp 2, 16-bit unsigned input, 8-bit output
    mon.start();
    wr(32'h004, 32'h1);
    wait_done("context", 32'h008, 400000);
    mon.report($sformatf("PV_%0drows", R));
    rd(32'h038, v); $display("    context: %0d cycles", v);

    bad = 0;
    for (int r = 0; r < R; r++)
      for (int d = 0; d < 64; d++) begin
        int off;
        logic [7:0] got;
        off = CTX + r * 64 + d;
        got = m_ew.mem[off / 32][8 * (off % 32) +: 8];
        if (got !== ctx_ref[r * 64 + d]) begin
          if (bad < 6) $display("FAIL %s ctx[%0d][%0d]: got %0d expected %0d", dir, r, d, $signed(got), $signed(ctx_ref[r * 64 + d]));
          bad++;
        end
      end
    checks++;
    if (bad != 0) begin errors++; $display("FAIL %s: %0d of %0d context values differ", dir, bad, R * 64); end
    else $display("  %s: %0d x 64 context values identical to the reference", dir, R);
  endtask

  initial begin
    logic [31:0] v;
    void'($urandom(32'h0C0DE));
    avs_address = 0; avs_write = 0; avs_read = 0; avs_writedata = 0; avs_byteenable = 4'hF;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);
    rd(32'h000, v); expect_eq("engine VERSION", v, 32'h4650_0002);
    rd(32'h100, v); expect_eq("softmax VERSION", v, 32'h534D_0002);
    rd(32'h1F0, v); expect_eq("unmapped softmax register", v, 32'hDEAD_BEEF);
    run_dir("vectors/attn_b0_h0");
    run_dir("vectors/attn_b11_h5");
    if (errors == 0) $display("PASS tb_attn_core: %0d checks", checks);
    else             $display("FAIL tb_attn_core: %0d errors in %0d checks", errors, checks);
    $finish;
  end

  initial begin
    #(2000000000);
    $display("FAIL tb_attn_core: global timeout");
    $finish;
  end
endmodule
