// Self-checking testbench for dma_rd against the random-wait Avalon memory model.
// Random addresses and lengths (including ones that cross 4 KiB), command back to
// back, and the timeout when the memory never answers.
`timescale 1ns/1ps

module tb_dma_rd;
  localparam int DEPTH = 4096;   // beats

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic         cmd_valid, cmd_ready, busy, done, err, beat_valid;
  logic [31:0]  cmd_addr;
  logic [15:0]  cmd_beats;
  logic [255:0] beat_data;
  logic [15:0]  beat_idx;

  logic [31:0]  m_address;
  logic         m_read, m_waitrequest, m_readdatavalid, hang;
  logic [7:0]   m_burstcount;
  logic [255:0] m_readdata;

  dma_rd #(.BURST(16), .TIMEOUT(200)) dut (.*);

  avalon_mem_model #(.DEPTH(DEPTH)) mem (
    .clk, .rst_n, .address(m_address), .read(m_read), .write(1'b0), .writedata('0),
    .byteenable('0), .burstcount(m_burstcount), .waitrequest(m_waitrequest),
    .readdata(m_readdata), .readdatavalid(m_readdatavalid), .hang_reads(hang)
  );

  int errors = 0, cmds = 0;
  int a, b;
  int got_beats;
  int expect_beats;
  logic [31:0] expect_addr;
  bit saw_done, saw_err;

  // check every streamed beat against the memory content
  always @(posedge clk) begin
    if (beat_valid) begin
      if (beat_idx !== got_beats[15:0]) begin
        if (errors < 10) $display("FAIL beat_idx %0d, expected %0d", beat_idx, got_beats);
        errors++;
      end
      if (beat_data !== mem.mem[expect_addr[31:5] + got_beats]) begin
        if (errors < 10) $display("FAIL data mismatch at beat %0d", got_beats);
        errors++;
      end
      got_beats++;
    end
    if (done) saw_done = 1;
    if (err)  saw_err  = 1;
  end

  // one command, waits for done/err
  task automatic run(input int addr_beat, input int beats, input bit expect_err);
    int to;
    got_beats = 0; expect_beats = beats; expect_addr = addr_beat * 32;
    saw_done = 0; saw_err = 0;
    @(posedge clk); #1;
    cmd_addr = addr_beat * 32; cmd_beats = beats; cmd_valid = 1;
    @(posedge clk); while (!cmd_ready) @(posedge clk);
    #1; cmd_valid = 0;
    to = 0;
    while (!saw_done && !saw_err && to < 100000) begin @(posedge clk); to++; end
    cmds++;
    if (expect_err) begin
      if (!saw_err) begin $display("FAIL expected a timeout error"); errors++; end
    end else begin
      if (!saw_done || saw_err) begin $display("FAIL cmd addr=%0d beats=%0d: done=%b err=%b", addr_beat, beats, saw_done, saw_err); errors++; end
      if (got_beats != expect_beats) begin $display("FAIL cmd addr=%0d beats=%0d: received %0d", addr_beat, beats, got_beats); errors++; end
    end
    repeat (3) @(posedge clk);
  endtask

  initial begin
    void'($urandom(32'hD3A0001));
    cmd_valid = 0; cmd_addr = 0; cmd_beats = 0; hang = 0;
    for (int i = 0; i < DEPTH; i++)
      for (int w = 0; w < 8; w++) mem.mem[i][32*w +: 32] = $urandom();
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    run(0, 1, 0);
    run(0, 16, 0);
    run(5, 17, 0);
    run(120, 20, 0);          // crosses the 4 KiB boundary at beat 128
    run(127, 2, 0);
    run(1, 300, 0);
    run(0, 0, 0);             // empty command
    for (int n = 0; n < 60; n++) begin
      a = $urandom_range(DEPTH - 400, 0);
      b = $urandom_range(300, 1);
      run(a, b, 0);
    end

    // timeout: memory swallows the request
    hang = 1;
    run(10, 4, 1);
    hang = 0;
    run(10, 4, 0);            // recovers

    if (mem.bad_access != 0) begin $display("FAIL %0d bad memory accesses (alignment/range/burst)", mem.bad_access); errors++; end
    if (errors == 0) $display("PASS tb_dma_rd: %0d commands", cmds);
    else             $display("FAIL tb_dma_rd: %0d errors", errors);
    $finish;
  end

  initial begin
    #50ms;
    $display("FAIL tb_dma_rd: global timeout");
    $finish;
  end
endmodule
