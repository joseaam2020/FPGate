// Self-checking testbench for gemm_engine over the random-wait Avalon memory model.
// It packs random matrices into memory with the H2 layouts (tile-major activations,
// per-tile weights + constants), runs a whole M x N x K operation and compares every
// output with a 64-bit reference. Covers 8/16-bit inputs (signed and unsigned), 8/16-bit
// outputs, M not a multiple of 16, several panels, back-to-back operations, bad
// parameters, a DMA read timeout and a hung write path (watchdog).
`timescale 1ns/1ps

module tb_gemm_engine;
  localparam int A_BEATS = 1024;
  localparam int W_BEATS = 64;
  localparam int A_BASE  = 32'h00000;
  localparam int W_BASE  = 32'h40000;
  localparam int O_BASE  = 32'h80000;
  localparam int DEPTH   = 1 << 15;         // beats = 1 MiB

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic         start, soft_reset;
  logic [31:0]  op, dim_m, dim_n, dim_k, addr_a, addr_w, addr_out, stride_out;
  logic         hw_done, hw_err_valid;
  logic [7:0]   hw_err_code;
  logic [31:0]  bytes_rd, bytes_wr;

  logic [31:0]  rd_address, wr_address, wr_byteenable;
  logic         rd_read, rd_waitrequest, rd_readdatavalid, wr_write, wr_waitrequest, hang_reads;
  logic [7:0]   rd_burstcount, wr_burstcount;
  logic [255:0] rd_readdata, wr_writedata;

  gemm_engine #(.A_BEATS(A_BEATS), .W_BEATS(W_BEATS), .LANES(4), .WD_LIMIT(20000), .DMA_TIMEOUT(300), .ADDR_LO(32'h0), .ADDR_HI(32'hFFFF_FFE0)) dut (.*);

  // one memory behind both masters: the read model serves reads, the write model serves
  // writes; both share the same backing array through hierarchical access in the checker
  avalon_mem_model #(.DEPTH(DEPTH)) mem_r (
    .clk, .rst_n, .address(rd_address), .read(rd_read), .write(1'b0), .writedata('0),
    .byteenable('0), .burstcount(rd_burstcount), .waitrequest(rd_waitrequest),
    .readdata(rd_readdata), .readdatavalid(rd_readdatavalid), .hang_reads(hang_reads)
  );
  logic block_writes;
  logic wr_wait_model;
  avalon_mem_model #(.DEPTH(DEPTH)) mem_w (
    .clk, .rst_n, .address(wr_address), .read(1'b0), .write(wr_write), .writedata(wr_writedata),
    .byteenable(wr_byteenable), .burstcount(wr_burstcount), .waitrequest(wr_wait_model),
    .readdata(), .readdatavalid(), .hang_reads(1'b0)
  );
  assign wr_waitrequest = wr_wait_model | block_writes;

  int errors = 0, ops = 0;

  // ---- reference data ----------------------------------------------------------
  int  x  [80][160];       // activations (full value)
  int  w  [64][160];       // weights int8
  int  bs [64];
  int unsigned m0v [64];
  int  shv [64];
  int  y  [80][64];        // expected output

  function automatic longint refq(input longint acc, input int unsigned mm, input int s, input bit wide);
    longint r, qm;
    r  = (acc * longint'(mm) + (64'sd1 << (s - 1))) >>> s;
    qm = wide ? 32767 : 127;
    if (r > qm) r = qm;
    if (r < -qm) r = -qm;
    return r;
  endfunction

  // copy of the written memory is read from mem_w.mem; reads come from mem_r.mem. Keep both in sync.
  task automatic mem_set(input int beat, input logic [255:0] v);
    mem_r.mem[beat] = v;
    mem_w.mem[beat] = v;
  endtask
  function automatic logic [255:0] mem_get_w(input int beat);
    return mem_w.mem[beat];
  endfunction

  logic [7:0] bytebuf [DEPTH*32];   // staging area, byte addressed

  task automatic load_memory(input int M, input int N, input int K, input bit in16, input bit sgn,
                             input bit out16, input int stride);
    int subs, planes, nt;
    subs = (M + 15) / 16; planes = in16 ? 2 : 1; nt = N / 16;
    // A: [subtile][plane][k][16 bytes]
    for (int s = 0; s < subs; s++)
      for (int p = 0; p < planes; p++)
        for (int k = 0; k < K; k++)
          for (int i = 0; i < 16; i++) begin
            int row, v;
            logic [7:0] b;
            row = s * 16 + i;
            v   = (row < M) ? x[row][k] : $urandom_range(255, 0);      // padding rows: junk
            if (!in16)       b = v[7:0];
            else if (p == 0) b = v[15:8];
            else             b = v[7:0];
            bytebuf[A_BASE + ((s * planes + p) * K + k) * 16 + i] = b;
          end
    // W: per tile of 16 channels: 8 beats of constants, then K words of 16 bytes
    for (int t = 0; t < nt; t++) begin
      int tbase;
      tbase = W_BASE + t * (256 + 16 * K);
      for (int j = 0; j < 16; j++) begin
        int n;
        n = t * 16 + j;
        for (int b = 0; b < 16; b++) bytebuf[tbase + 16 * j + b] = 8'h00;
        for (int b = 0; b < 4; b++) bytebuf[tbase + 16 * j + b]     = bs[n][8*b +: 8];
        for (int b = 0; b < 4; b++) bytebuf[tbase + 16 * j + 4 + b] = m0v[n][8*b +: 8];
        bytebuf[tbase + 16 * j + 8] = 8'(shv[n]);
      end
      for (int k = 0; k < K; k++)
        for (int j = 0; j < 16; j++)
          bytebuf[tbase + 256 + 16 * k + j] = 8'(w[t * 16 + j][k]);
    end
    // output area pattern
    for (int a = 0; a < (M + 16) * stride; a++) bytebuf[O_BASE + a] = 8'hCC;
    // push staging bytes into both memory copies
    for (int bt = 0; bt < DEPTH; bt++) begin
      logic [255:0] v;
      bit touched;
      touched = (bt * 32 < 32'h10000) || (bt * 32 >= W_BASE && bt * 32 < W_BASE + 32'h4000) ||
                (bt * 32 >= O_BASE && bt * 32 < O_BASE + (M + 16) * stride + 32);
      if (touched) begin
        for (int b = 0; b < 32; b++) v[8*b +: 8] = bytebuf[bt * 32 + b];
        mem_set(bt, v);
      end
    end
  endtask

  task automatic compute_reference(input int M, input int N, input int K, input bit out16);
    for (int r = 0; r < M; r++)
      for (int n = 0; n < N; n++) begin
        longint acc;
        acc = bs[n];
        for (int k = 0; k < K; k++) acc += longint'(x[r][k]) * longint'(w[n][k]);
        y[r][n] = int'(refq(acc, m0v[n], shv[n], out16));
      end
  endtask

  task automatic gen_data(input int M, input int N, input int K, input bit in16, input bit sgn);
    for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++) begin
        if (!in16)     x[r][k] = $urandom_range(255, 0) - 128;
        else if (sgn)  x[r][k] = $urandom_range(65535, 0) - 32768;
        else           x[r][k] = $urandom_range(65535, 0);
      end
    for (int n = 0; n < N; n++) begin
      for (int k = 0; k < K; k++) w[n][k] = $urandom_range(255, 0) - 128;
      bs[n]  = $urandom_range(2000, 0) - 1000;
      m0v[n] = 32'h4000_0000 + $urandom_range(32'h3fff_ffff, 0);
      shv[n] = $urandom_range(in16 ? 46 : 36, in16 ? 30 : 24);       // mixes saturated and plain values
    end
  endtask

  // ---- run one operation ----------------------------------------------------------
  bit got_done, got_err;
  logic [7:0] got_code;
  always @(posedge clk) begin
    if (hw_done) got_done = 1;
    if (hw_err_valid) begin got_err = 1; got_code = hw_err_code; end
  end

  task automatic launch(input int M, input int N, input int K, input bit in16, input bit sgn,
                        input bit out16, input int spp, input int stride);
    @(posedge clk); #1;
    op = {16'b0, 8'(spp), 5'b0, out16, sgn, in16};
    dim_m = M; dim_n = N; dim_k = K;
    addr_a = A_BASE; addr_w = W_BASE; addr_out = O_BASE; stride_out = stride;
    got_done = 0; got_err = 0; got_code = 0;
    start = 1;
    @(posedge clk); #1;
    start = 0;
  endtask

  task automatic wait_end(input int max_cycles, output int cycles);
    cycles = 0;
    while (!got_done && !got_err && cycles < max_cycles) begin @(posedge clk); cycles++; end
    repeat (4) @(posedge clk);
  endtask

  task automatic run_case(input string name, input int M, input int N, input int K, input bit in16,
                          input bit sgn, input bit out16, input int spp);
    int stride, cyc, panels, subs, planes, nt, exp_rd, exp_wr, bad;
    int panel_sub, left;
    stride = ((N * (out16 ? 2 : 1) + 31) / 32) * 32 + 32 * $urandom_range(2, 0);
    gen_data(M, N, K, in16, sgn);
    load_memory(M, N, K, in16, sgn, out16, stride);
    compute_reference(M, N, K, out16);
    launch(M, N, K, in16, sgn, out16, spp, stride);
    wait_end(2000000, cyc);
    ops++;
    if (!got_done || got_err) begin
      $display("FAIL %s: done=%b err=%b code=%0d", name, got_done, got_err, got_code);
      errors++;
      return;
    end
    // outputs
    bad = 0;
    for (int r = 0; r < M; r++)
      for (int n = 0; n < N; n++) begin
        logic [255:0] beatv;
        int byte_off, got;
        byte_off = O_BASE + r * stride + n * (out16 ? 2 : 1);
        beatv = mem_get_w(byte_off / 32);
        if (out16) got = int'(signed'(beatv[8 * (byte_off % 32) +: 16]));
        else       got = int'(signed'(beatv[8 * (byte_off % 32) +: 8]));
        if (got !== y[r][n]) begin
          if (bad < 5) $display("FAIL %s [%0d][%0d]: got %0d expected %0d", name, r, n, got, y[r][n]);
          bad++;
        end
      end
    // rows beyond M must be untouched
    for (int r = M; r < ((M + 15) / 16) * 16; r++)
      for (int b = 0; b < N * (out16 ? 2 : 1); b++) begin
        logic [255:0] beatv;
        int byte_off;
        byte_off = O_BASE + r * stride + b;
        beatv = mem_get_w(byte_off / 32);
        if (beatv[8 * (byte_off % 32) +: 8] !== 8'hCC) begin
          if (bad < 5) $display("FAIL %s: padding row %0d was written", name, r);
          bad++;
          break;
        end
      end
    // byte counters
    planes = in16 ? 2 : 1; subs = (M + 15) / 16; nt = N / 16;
    exp_rd = 0; left = subs; panels = 0;
    while (left > 0) begin
      panel_sub = (left < spp) ? left : spp;
      exp_rd += panel_sub * planes * (K / 2) + nt * (8 + K / 2);
      left -= panel_sub; panels++;
    end
    exp_rd *= 32;
    exp_wr = M * nt * (out16 ? 32 : 16);
    if (bytes_rd !== exp_rd) begin $display("FAIL %s: BYTES_RD %0d expected %0d", name, bytes_rd, exp_rd); bad++; end
    if (bytes_wr !== exp_wr) begin $display("FAIL %s: BYTES_WR %0d expected %0d", name, bytes_wr, exp_wr); bad++; end
    if (bad != 0) errors++;
  endtask

  task automatic expect_error(input string name, input int M, input int N, input int K, input bit in16,
                              input int spp, input byte code);
    int cyc;
    launch(M, N, K, in16, 1'b1, 1'b1, spp, 128);
    wait_end(200000, cyc);
    ops++;
    if (!got_err || got_code !== code) begin
      $display("FAIL %s: expected error %0d, got err=%b code=%0d done=%b", name, code, got_err, got_code, got_done);
      errors++;
    end
  endtask

  int M, N, K, spp;
  bit in16, sgn, out16;

  initial begin
    void'($urandom(32'hE96A0003));
    start = 0; soft_reset = 0; op = 0; dim_m = 0; dim_n = 0; dim_k = 0;
    addr_a = 0; addr_w = 0; addr_out = 0; stride_out = 0; hang_reads = 0; block_writes = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    // directed shapes
    run_case("8-bit 16x16xK16",  16, 16, 16, 0, 1, 0, 2);
    run_case("8-bit out16 17x32xK32", 17, 32, 32, 0, 1, 1, 2);
    run_case("16-bit signed 20x16xK24 out16", 20, 16, 24, 1, 1, 1, 2);
    run_case("16-bit unsigned 33x32xK20", 33, 32, 20, 1, 0, 0, 3);
    run_case("M=1", 1, 16, 8, 0, 1, 1, 1);
    run_case("M=65 several panels", 65, 16, 12, 0, 1, 1, 2);
    run_case("N=64, spp=1", 40, 64, 16, 1, 1, 1, 1);

    // random shapes
    for (int n = 0; n < 24; n++) begin
      M     = $urandom_range(70, 1);
      N     = 16 * $urandom_range(4, 1);
      K     = 2 * $urandom_range(64, 1);
      in16  = $urandom_range(1, 0);
      sgn   = in16 ? $urandom_range(1, 0) : 1;
      out16 = $urandom_range(1, 0);
      spp   = $urandom_range(in16 ? 4 : 8, 1);
      run_case($sformatf("random %0d: M=%0d N=%0d K=%0d in16=%0d sgn=%0d out16=%0d spp=%0d", n, M, N, K, in16, sgn, out16, spp),
               M, N, K, in16, sgn, out16, spp);
    end

    // bad parameters
    expect_error("K odd", 16, 16, 7, 0, 2, 8'd2);
    expect_error("N not multiple of 16", 16, 20, 8, 0, 2, 8'd2);
    expect_error("M zero", 0, 16, 8, 0, 2, 8'd2);
    expect_error("panel does not fit the A RAM", 16, 16, 128, 1, 200, 8'd2);
    expect_error("K larger than weight buffer", 16, 16, 200, 0, 2, 8'd2);

    // read timeout, then recovery on the very next operation
    hang_reads = 1;
    gen_data(16, 16, 16, 0, 1); load_memory(16, 16, 16, 0, 1, 1, 128); compute_reference(16, 16, 16, 1);
    launch(16, 16, 16, 0, 1, 1, 2, 128);
    begin int c; wait_end(100000, c); end
    ops++;
    if (!got_err || got_code !== 8'd3) begin $display("FAIL read timeout: err=%b code=%0d", got_err, got_code); errors++; end
    hang_reads = 0;
    run_case("after read timeout", 16, 16, 16, 0, 1, 1, 2);

    // write path that never answers: DMA write timeout (code 4) or watchdog
    block_writes = 1;
    gen_data(16, 16, 16, 0, 1); load_memory(16, 16, 16, 0, 1, 1, 128);
    launch(16, 16, 16, 0, 1, 1, 2, 128);
    begin int c; wait_end(200000, c); end
    ops++;
    if (!got_err || (got_code !== 8'd4 && got_code !== 8'd6)) begin
      $display("FAIL write stall: err=%b code=%0d", got_err, got_code); errors++;
    end
    block_writes = 0;
    run_case("after write stall", 16, 16, 16, 0, 1, 1, 2);

    if (mem_r.bad_access != 0 || mem_w.bad_access != 0) begin
      $display("FAIL: bad memory accesses r=%0d w=%0d", mem_r.bad_access, mem_w.bad_access);
      errors++;
    end
    if (errors == 0) $display("PASS tb_gemm_engine: %0d operations", ops);
    else             $display("FAIL tb_gemm_engine: %0d errors in %0d operations", errors, ops);
    $finish;
  end

  initial begin
    #2s;
    $display("FAIL tb_gemm_engine: global timeout");
    $finish;
  end
endmodule
