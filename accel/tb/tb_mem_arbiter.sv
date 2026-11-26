// mem_arbiter: four masters with random traffic against (0) avalon_mem_model (random waits, gaps, latency,
// single-beat writes) and (1) a pipelined slave with deep read latency (to fill the 16 outstanding reads) and
// multi-beat write bursts, which must reach the slave without another master's beats in between.
// Checks: read data routed to the right master and in order, memory contents after the writes (byte enables
// included), burst integrity, never more than 16 reads outstanding at the slave (and that 16 is reached in
// mode 1), per-master ordering. Prints PASS / FAIL.
`timescale 1ns/1ps

module arb_harness #(
  parameter int MODE   = 0,      // 0: avalon_mem_model, 1: pipelined slave with bursts
  parameter int NRD    = 300,    // read commands per read master
  parameter int NWR    = 300     // write bursts per write master
)(
  input  logic clk,
  input  logic rst_n,
  output logic done,
  output int   errors,
  output int   max_out
);
  localparam int DEPTH = 4096;   // beats; reads from 0..1023, writer 1 in 1024..2047, writer 3 in 2048..3071

  logic [31:0]  m_address [4];
  logic         m_read [4], m_write [4], m_waitrequest [4], m_readdatavalid [4];
  logic [255:0] m_writedata [4];
  logic [31:0]  m_byteenable [4];
  logic [7:0]   m_burstcount [4];
  logic [255:0] m_readdata;

  logic [31:0]  s_address, s_byteenable;
  logic         s_read, s_write, s_waitrequest, s_readdatavalid;
  logic [255:0] s_writedata, s_readdata;
  logic [7:0]   s_burstcount;

  mem_arbiter u_arb (.*);
  // outstanding reads as the arbiter counts them (includes the two commands still in its output stages)
  int arb_max = 0;
  always @(posedge clk) if (int'(u_arb.id_cnt) > arb_max) arb_max = u_arb.id_cnt;
  assign max_out = arb_max;

  function automatic logic [255:0] pat(input int unsigned a);
    logic [255:0] d;
    for (int j = 0; j < 8; j++) d[32*j +: 32] = a * 32'h9E3779B1 + j * 32'h01000193;
    return d;
  endfunction

  function automatic logic [255:0] mkdata(input int id, input int serial, input int beat);
    logic [255:0] d;
    for (int j = 0; j < 8; j++) d[32*j +: 32] = (beat * 32'h85EBCA6B) ^ (j * 32'h27D4EB2F) ^ (id << 28);
    d[255:224] = {serial[29:0], 2'(id)};
    return d;
  endfunction

  // ---- slave ----------------------------------------------------------------------------------------------
  logic [255:0] smem [DEPTH];
  event snap_ev;
  initial for (int i = 0; i < DEPTH; i++) smem[i] = (i < 1024) ? pat(i) : '0;

  if (MODE == 0) begin : g_model
    avalon_mem_model #(.DEPTH(DEPTH), .WAIT_PCT(30), .GAP_PCT(30), .LAT(5)) u_mem (
      .clk, .rst_n, .address(s_address), .read(s_read), .write(s_write), .writedata(s_writedata),
      .byteenable(s_byteenable), .burstcount(s_burstcount), .waitrequest(s_waitrequest),
      .readdata(s_readdata), .readdatavalid(s_readdatavalid), .hang_reads(1'b0));
    initial begin
      for (int i = 0; i < DEPTH; i++) u_mem.mem[i] = smem[i];
    end
    always @(snap_ev) for (int i = 0; i < DEPTH; i++) smem[i] = u_mem.mem[i];
  end else begin : g_pipe
    logic rnd;
    always @(posedge clk) rnd <= ($urandom_range(99, 0) < 20);
    assign s_waitrequest = !rst_n || rnd;

    typedef struct { int addr; int bc; int ready; } rq_t;
    rq_t rq [$];
    int  cyc = 0, k = 0, wr_left = 0, wr_addr = 0;
    logic [31:0] tag;
    always @(posedge clk) begin
      cyc++;
      s_readdatavalid <= 1'b0;
      if (rst_n) begin
        if (!s_waitrequest && s_write) begin
          if (wr_left == 0) begin wr_addr = s_address >> 5; wr_left = s_burstcount; tag = s_writedata[255:224]; end
          else if (s_writedata[255:224] !== tag) begin
            errors++; $display("FAIL: write burst interleaved (tag %08x, beat tag %08x)", tag, s_writedata[255:224]);
          end
          if (s_address >> 5 != wr_addr) begin errors++; $display("FAIL: write address %0d, expected %0d", s_address >> 5, wr_addr); end
          for (int b = 0; b < 32; b++) if (s_byteenable[b]) smem[wr_addr][8*b +: 8] = s_writedata[8*b +: 8];
          wr_addr++; wr_left--;
        end
        if (!s_waitrequest && s_read) begin
          rq.push_back('{addr: s_address >> 5, bc: int'(s_burstcount), ready: cyc + 40 + $urandom_range(20, 0)});
          if (rq.size() > 16) begin errors++; $display("FAIL: %0d reads outstanding at the slave", rq.size()); end
        end
        if (rq.size() > 0 && cyc >= rq[0].ready && $urandom_range(99, 0) < 80) begin
          s_readdata      <= pat(rq[0].addr + k);
          s_readdatavalid <= 1'b1;
          k++;
          if (k == rq[0].bc) begin k = 0; void'(rq.pop_front()); end
        end
      end
    end
  end

  // ---- read masters (ports 0 and 2) ---------------------------------------------------------------------------
  typedef struct { int addr; int bc; } exp_t;
  exp_t expq [4][$];
  int   beat_no [4];
  int   rd_issued [4];
  logic wr_done [4];

  always @(posedge clk) if (rst_n) begin
    for (int p = 0; p < 4; p += 2) begin
      if (m_readdatavalid[p]) begin
        if (expq[p].size() == 0) begin errors++; $display("FAIL: port %0d got data nobody asked for", p); end
        else begin
          if (m_readdata !== pat(expq[p][0].addr + beat_no[p])) begin
            errors++; if (errors < 10) $display("FAIL: port %0d beat %0d of burst @%0d wrong", p, beat_no[p], expq[p][0].addr);
          end
          beat_no[p]++;
          if (beat_no[p] == expq[p][0].bc) begin beat_no[p] = 0; void'(expq[p].pop_front()); end
        end
      end
    end
    // a data beat for a port that is not a read master would show up here
    if (m_readdatavalid[1] || m_readdatavalid[3]) begin errors++; $display("FAIL: readdatavalid on a write port"); end
  end

  initial begin
    for (int p = 0; p < 4; p++) begin
      m_read[p] = 0; m_write[p] = 0; m_address[p] = 0; m_writedata[p] = '0; m_byteenable[p] = '0; m_burstcount[p] = 0;
      beat_no[p] = 0; rd_issued[p] = 0; wr_done[p] = 0;
    end
    errors = 0;
  end

  task automatic read_master(input int p);
    int bc, a;
    wait (rst_n);
    repeat (NRD) begin
      bc = (MODE == 1) ? $urandom_range(4, 1) : $urandom_range(16, 1);
      a  = $urandom_range(1023 - bc, 0);
      @(posedge clk); #1;
      m_address[p] = a * 32; m_burstcount[p] = bc; m_read[p] = 1;
      expq[p].push_back('{addr: a, bc: bc});
      do @(posedge clk); while (m_waitrequest[p]);
      #1 m_read[p] = 0;
      rd_issued[p]++;
      if ($urandom_range(3, 0) == 0) repeat ($urandom_range(5, 1)) @(posedge clk);
    end
  endtask

  logic [255:0] expmem [int];
  task automatic write_master(input int p, input int base);
    int bc, serial, cur, k;
    logic [31:0] be;
    logic [255:0] d, mask;
    cur = base; serial = 0;
    wait (rst_n);
    repeat (NWR) begin
      bc = (MODE == 1) ? $urandom_range(4, 1) : 1;
      serial++;
      for (k = 0; k < bc; k++) begin
        if (k > 0 && $urandom_range(3, 0) == 0) begin   // idle cycles inside the burst: the bus must stay locked
          #1 m_write[p] = 0;
          repeat ($urandom_range(4, 1)) @(posedge clk);
        end
        @(posedge clk); #1;
        d  = mkdata(p, serial * 4 + p, cur + k);
        be = ($urandom_range(3, 0) == 0) ? $urandom : 32'hFFFF_FFFF;
        if (be == 0) be = 32'h1;
        for (int b = 0; b < 32; b++) mask[8*b +: 8] = {8{be[b]}};
        expmem[cur + k] = (expmem.exists(cur + k) ? expmem[cur + k] : '0) & ~mask | d & mask;
        m_address[p] = (cur + k) * 32; m_writedata[p] = d; m_byteenable[p] = be; m_burstcount[p] = bc; m_write[p] = 1;
        do @(posedge clk); while (m_waitrequest[p]);
        #1 m_write[p] = 0;
      end
      cur += bc;
      if ($urandom_range(3, 0) == 0) repeat ($urandom_range(5, 1)) @(posedge clk);
    end
    wr_done[p] = 1;
  endtask

  initial begin
    done = 0;
    fork
      read_master(0);
      read_master(2);
      write_master(1, 1024);
      write_master(3, 2048);
    join
    repeat (400) @(posedge clk);
    for (int p = 0; p < 4; p += 2)
      if (expq[p].size() != 0) begin errors++; $display("FAIL: port %0d has %0d reads never answered", p, expq[p].size()); end
    -> snap_ev; #1;
    foreach (expmem[b])
      if (smem[b] !== expmem[b]) begin
        errors++; if (errors < 10) $display("FAIL: memory beat %0d differs", b);
      end
    done = 1;
  end
endmodule

module tb_mem_arbiter;
  logic clk = 0, rst_n = 0;
  always #3.333 clk = ~clk;

  logic d0, d1;
  int   e0, e1, mo0, mo1;
  arb_harness #(.MODE(0)) h0 (.clk, .rst_n, .done(d0), .errors(e0), .max_out(mo0));
  arb_harness #(.MODE(1)) h1 (.clk, .rst_n, .done(d1), .errors(e1), .max_out(mo1));

  initial begin
    repeat (10) @(posedge clk);
    rst_n = 1;
    fork
      begin wait (d0 && d1); end
      begin repeat (400000) @(posedge clk); $display("FAIL: timeout"); $finish; end
    join_any
    $display("RESULT mem_arbiter: mode0 errors=%0d max_out=%0d, mode1 errors=%0d max_out=%0d", e0, mo0, e1, mo1);
    if (e0 == 0 && e1 == 0 && mo1 == 16) $display("PASS tb_mem_arbiter");
    else $display("FAIL tb_mem_arbiter");
    $finish;
  end
endmodule
