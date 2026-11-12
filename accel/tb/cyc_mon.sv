// Per-state cycle counters for the testbenches, read by hierarchical reference (nothing in rtl/ changes).
// Instantiate inside a testbench that has the core as `dut`; start() clears and enables the counters,
// report() stops them and prints one line:  RESULT cyc: <op> <unit>.<state>=<cycles> ...
// Sequencer states: S_RUN = array computing, S_DRAIN = array emptying.
`timescale 1ns/1ps

module cyc_mon #(parameter bit SEQ = 1, parameter bit SM = 1);
  localparam string SEQ_NAMES [19] = '{"IDLE", "CHK", "CHK_LOOP", "PSTART", "ACALC", "LOADA_CMD", "LOADA_WAIT",
    "LOADW_CMD", "LOADW_WAIT", "TSTART", "PF_CMD", "SSTART", "RUN", "DRAIN", "TEND", "PEND", "FLUSH", "DONE", "ERR"};
  localparam string SM_NAMES [15] = '{"IDLE", "CHK", "REQ", "RD", "MAXD", "P2", "P2D", "RCP", "P3", "P3D",
    "NEXT", "OUT", "OUTD", "FIN", "ERR"};

  bit  en = 0;
  int  seq_cnt [19];
  int  sm_cnt  [15];
  int  total;

  always @(posedge dut.clk) if (en) begin
    total++;
    seq_cnt[int'(dut.u_engine.u_seq.state)]++;
    sm_cnt[int'(dut.u_softmax.state)]++;
  end

  task automatic start();
    foreach (seq_cnt[i]) seq_cnt[i] = 0;
    foreach (sm_cnt[i])  sm_cnt[i] = 0;
    total = 0;
    en = 1;
  endtask

  task automatic report(input string op);
    string l;
    en = 0;
    l = $sformatf("RESULT cyc: %s total=%0d", op, total);
    if (SEQ) foreach (seq_cnt[i]) if (seq_cnt[i] != 0 && i != 0) l = {l, $sformatf(" seq.%s=%0d", SEQ_NAMES[i], seq_cnt[i])};
    if (SM)  foreach (sm_cnt[i])  if (sm_cnt[i]  != 0 && i != 0) l = {l, $sformatf(" sm.%s=%0d", SM_NAMES[i], sm_cnt[i])};
    $display("%s", l);
  endtask
endmodule
