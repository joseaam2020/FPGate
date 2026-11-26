// Softmax unit against the reference: real attention scores of the e1_r7 model (block 0 head 0 and block 11
// head 5, tools/pack_attn_softmax.py) go through attn_softmax with a simulated DDR3 (random waits and gaps).
// P must equal the reference probabilities bit by bit, in the activation format of the engine
// ([subtile][plane][k][16 bytes]); the padding column (k = 1025) and the rows past the last row are zero.
// The run directory is accel/ (sim.ps1 / sim_tb.ps1 start there).
`timescale 1ns/1ps

module tb_attn_softmax;
  localparam int DEPTH = 1 << 13;
  localparam int KP    = 1026;

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic         start = 0, soft_reset = 0;
  logic [15:0]  rows, cols, kp;
  logic [31:0]  s_addr, s_stride, p_addr;
  logic [30:0]  sm_m0;
  logic [5:0]   sm_sh;
  logic         busy, done, err_valid;
  logic [7:0]   err_code;
  logic [31:0]  bytes_rd, bytes_wr;

  logic [31:0]  rd_address, wr_address, wr_byteenable;
  logic         rd_read, rd_waitrequest, rd_readdatavalid, wr_write, wr_waitrequest;
  logic [7:0]   rd_burstcount, wr_burstcount;
  logic [255:0] rd_readdata, wr_writedata;

  attn_softmax #(.LANES2(4), .ROW_WORDS(130), .ADDR_LO(32'h0), .ADDR_HI(32'hFFFF_FFE0)) dut (.*);

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
  logic [255:0] s_img [32 * 130];
  logic [15:0]  p_ref [32 * 1025];

  task automatic run_dir(input string dir, input int R);
    int fd, rc, pr, pc, pstride, to;
    longint m0v, shv;
    int bad, zero_bad;
    $readmemh({dir, "/sm_s.hex"}, s_img);
    $readmemh({dir, "/sm_p.hex"}, p_ref);
    fd = $fopen({dir, "/sm_params.txt"}, "r");
    rc = $fscanf(fd, "%d %d %d %d %d", pr, pc, pstride, m0v, shv);
    $fclose(fd);

    for (int i = 0; i < DEPTH; i++) begin mem_r.mem[i] = '0; mem_w.mem[i] = '0; end
    for (int i = 0; i < R * 130; i++) mem_r.mem[i] = s_img[i];
    rows = R; cols = 1025; kp = KP;
    s_addr = 0; s_stride = pstride; p_addr = 0;
    sm_m0 = 31'(m0v); sm_sh = 6'(shv);
    @(posedge clk); #1;
    start = 1;
    @(posedge clk); #1;
    start = 0;
    to = 0;
    while (!done && !err_valid && to < 3000000) begin @(posedge clk); to++; end
    checks++;
    if (err_valid) begin $display("FAIL %s: error code %0d", dir, err_code); errors++; end
    else if (!done) begin $display("FAIL %s: timeout", dir); errors++; end
    else $display("  %s: done after %0d cycles, bytes read %0d, written %0d", dir, to, bytes_rd, bytes_wr);

    bad = 0; zero_bad = 0;
    for (int r = 0; r < ((R + 15) / 16) * 16; r++)
      for (int k = 0; k < KP; k++) begin
        int sub, i, base_hi, base_lo;
        logic [7:0] hi, lo;
        logic [15:0] got, exp;
        sub = r / 16; i = r % 16;
        base_hi = sub * 2 * KP * 16 + k * 16 + i;
        base_lo = sub * 2 * KP * 16 + KP * 16 + k * 16 + i;
        hi = mem_w.mem[base_hi / 32][8 * (base_hi % 32) +: 8];
        lo = mem_w.mem[base_lo / 32][8 * (base_lo % 32) +: 8];
        got = {hi, lo};
        exp = (r < R && k < 1025) ? p_ref[r * 1025 + k] : 16'd0;
        if (got !== exp) begin
          if (bad < 6) $display("FAIL %s row %0d k %0d: got %0d expected %0d", dir, r, k, got, exp);
          bad++;
          if (r >= R || k >= 1025) zero_bad++;
        end
      end
    checks++;
    if (bad != 0) begin errors++; $display("FAIL %s: %0d probabilities differ (%0d in the padding)", dir, bad, zero_bad); end
    else $display("  %s: %0d rows of P identical to the reference (padding and unused rows are zero)", dir, R);
    checks++;
    if (bytes_wr !== ((R + 15) / 16) * 2 * (KP / 2) * 32) begin
      errors++; $display("FAIL %s: bytes_wr = %0d", dir, bytes_wr);
    end
  endtask

  initial begin
    void'($urandom(32'h0C0DE));
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);
    run_dir("vectors/attn_b0_h0", 32);
    run_dir("vectors/attn_b11_h5", 20);

    if (errors == 0) $display("PASS tb_attn_softmax: %0d checks", checks);
    else             $display("FAIL tb_attn_softmax: %0d errors in %0d checks", errors, checks);
    $finish;
  end

  initial begin
    #(2000000000);
    $display("FAIL tb_attn_softmax: global timeout");
    $finish;
  end
endmodule
