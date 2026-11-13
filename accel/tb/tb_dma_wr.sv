// Self-checking testbench for dma_wr against the random-wait Avalon memory model:
// random beats with random byte enables, compared with a shadow copy of the memory.
`timescale 1ns/1ps

module tb_dma_wr;
  localparam int DEPTH = 1024;

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic         flush = 1'b0;
  logic         in_valid, in_ready, busy, wr_beat, err, err_win;
  logic [31:0]  in_addr, in_be;
  logic [255:0] in_data;

  logic [31:0]  m_address, m_byteenable;
  logic         m_write, m_waitrequest;
  logic [255:0] m_writedata;
  logic [7:0]   m_burstcount;

  dma_wr #(.TIMEOUT(200), .ADDR_LO(32'h0), .ADDR_HI(32'hFFFF_FFE0)) dut (.*);

  avalon_mem_model #(.DEPTH(DEPTH)) mem (
    .clk, .rst_n, .address(m_address), .read(1'b0), .write(m_write), .writedata(m_writedata),
    .byteenable(m_byteenable), .burstcount(m_burstcount), .waitrequest(m_waitrequest),
    .readdata(), .readdatavalid(), .hang_reads(1'b0)
  );

  logic [255:0] shadow [DEPTH];
  int errors = 0, beats_sent = 0, beats_done = 0;

  always @(posedge clk) if (wr_beat) beats_done++;

  task automatic send(input int beat, input logic [255:0] d, input logic [31:0] be);
    @(posedge clk); #1;
    in_valid = 1; in_addr = beat * 32; in_data = d; in_be = be;
    @(posedge clk);
    while (!in_ready) @(posedge clk);
    #1; in_valid = 0;
    for (int b = 0; b < 32; b++) if (be[b]) shadow[beat][8*b +: 8] = d[8*b +: 8];
    beats_sent++;
  endtask

  logic [255:0] d;
  logic [31:0]  be;
  int to, n, beat;

  initial begin
    void'($urandom(32'h0D3A0002));
    in_valid = 0; in_addr = 0; in_data = 0; in_be = 0;
    for (int i = 0; i < DEPTH; i++) begin
      for (int w = 0; w < 8; w++) mem.mem[i][32*w +: 32] = 32'hA5A5A5A5;
      shadow[i] = mem.mem[i];
    end
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    // full 32-byte beats, 16-byte halves (8-bit outputs), random byte enables
    for (n = 0; n < 400; n++) begin
      for (int w = 0; w < 8; w++) d[32*w +: 32] = $urandom();
      beat = $urandom_range(DEPTH - 1, 0);
      case ($urandom_range(3, 0))
        0: be = 32'hFFFF_FFFF;
        1: be = 32'h0000_FFFF;
        2: be = 32'hFFFF_0000;
        default: be = $urandom();
      endcase
      send(beat, d, be);
    end

    to = 0;
    while ((busy || beats_done < beats_sent) && to < 10000) begin @(posedge clk); to++; end
    repeat (4) @(posedge clk);

    if (beats_done != beats_sent) begin $display("FAIL: %0d beats sent, %0d completed", beats_sent, beats_done); errors++; end
    for (int i = 0; i < DEPTH; i++)
      if (mem.mem[i] !== shadow[i]) begin
        if (errors < 10) $display("FAIL: memory beat %0d differs", i);
        errors++;
      end
    if (mem.bad_access != 0) begin $display("FAIL: %0d bad accesses", mem.bad_access); errors++; end

    if (errors == 0) $display("PASS tb_dma_wr: %0d beats", beats_sent);
    else             $display("FAIL tb_dma_wr: %0d errors", errors);
    $finish;
  end

  initial begin
    #50ms;
    $display("FAIL tb_dma_wr: global timeout");
    $finish;
  end
endmodule
