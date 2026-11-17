// Fused SwiGLU on fc1 (OP.SWIGLU of gemm_engine, swiglu_stage), bit by bit.
//
// The engine runs identity "projections" so that exact INT8 values reach the epilogue: A = [gi | ui] (K = 2 * NOUT),
// the weights are interleaved tiles gate(0), up(0), gate(1), up(1), ... whose tile picks one column of A
// (bias 0, M0 = 2^30, shift 30: y = x). The engine must write h = swiglu(gi, ui) of INT16, one tile per gate / up pair.
//   * Real data: blocks 0 and 11 of image 0 (vectors/<b>_swiglu from tools/dump_swiglu.py: fc1g / fc1x outputs,
//     the SiLU table, M0 / shift and the expected swiglu_i of the package), 32 rows x 32 neurons.
//   * Synthetic: random table, inputs, M0 and shifts (30..42, and a few extremes to saturate), M not a multiple of 16,
//     panels of 1, 3 and 16 subtiles, write path stalled in bursts.
//   * The output beyond the h tiles stays untouched (gate rows write nothing), BYTES_WR counts the h rows only.
//   * Bad combinations (16-bit output, odd tile count, more than 16 subtiles per panel) end in error 2.
`timescale 1ns/1ps

module tb_swiglu_fused;
  localparam int A_BEATS = 4096;
  localparam int W_BEATS = 1024;
  localparam int A_BASE  = 32'h00000;
  localparam int W_BASE  = 32'h40000;
  localparam int O_BASE  = 32'h60000;
  localparam int DEPTH   = 1 << 15;          // 1 MiB

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic         start = 0, soft_reset = 0;
  logic [31:0]  op = 0, dim_m = 0, dim_n = 0, dim_k = 0, addr_a = 0, addr_w = 0, addr_out = 0, stride_out = 0;
  logic [31:0]  out_tstride = 0, head_tiles = 0, head_stride = 0, row0 = 0, rope_cfg = 0, rope_tab_data = 0;
  logic         rope_tab_we = 0;
  logic [31:0]  sw_m0 = 0, sw_shift = 0, sw_tab_data = 0;
  logic         sw_tab_we = 0;
  logic         hw_done, hw_err_valid;
  logic [7:0]   hw_err_code;
  logic [31:0]  bytes_rd, bytes_wr;

  logic [31:0]  rd_address, wr_address, wr_byteenable;
  logic         rd_read, rd_waitrequest, rd_readdatavalid, wr_write, wr_waitrequest;
  logic [7:0]   rd_burstcount, wr_burstcount;
  logic [255:0] rd_readdata, wr_writedata;

  gemm_engine #(.A_BEATS(A_BEATS), .W_BEATS(W_BEATS), .LANES(4), .ADDR_LO(32'h0), .ADDR_HI(32'hFFFF_FFE0)) dut (.*);

  avalon_mem_model #(.DEPTH(DEPTH)) mem_r (
    .clk, .rst_n, .address(rd_address), .read(rd_read), .write(1'b0), .writedata('0),
    .byteenable('0), .burstcount(rd_burstcount), .waitrequest(rd_waitrequest),
    .readdata(rd_readdata), .readdatavalid(rd_readdatavalid), .hang_reads(1'b0)
  );
  logic block_writes = 0;
  logic wr_wait_model;
  avalon_mem_model #(.DEPTH(DEPTH)) mem_w (
    .clk, .rst_n, .address(wr_address), .read(1'b0), .write(wr_write), .writedata(wr_writedata),
    .byteenable(wr_byteenable), .burstcount(wr_burstcount), .waitrequest(wr_wait_model),
    .readdata(), .readdatavalid(), .hang_reads(1'b0)
  );
  assign wr_waitrequest = wr_wait_model | block_writes;

  bit stall_en = 0;
  initial forever begin
    @(posedge clk);
    if (stall_en && $urandom_range(999, 0) < 4) begin
      block_writes = 1;
      repeat ($urandom_range(600, 60)) @(posedge clk);
      block_writes = 0;
      repeat ($urandom_range(300, 20)) @(posedge clk);
    end
  end

  int errors = 0, runs = 0;
  bit got_done, got_err;
  logic [7:0] got_code;
  always @(posedge clk) begin
    if (hw_done) got_done = 1;
    if (hw_err_valid) begin got_err = 1; got_code = hw_err_code; end
  end

  function automatic logic [7:0] peek(input int a);
    return mem_w.mem[a >> 5][8 * (a & 31) +: 8];
  endfunction
  task automatic poke(input int a, input logic [7:0] b);
    mem_r.mem[a >> 5][8 * (a & 31) +: 8] = b;
    mem_w.mem[a >> 5][8 * (a & 31) +: 8] = b;
  endtask

  // ---- data of the run ---------------------------------------------------------------------------------------
  int  gv [320][64];                          // gate values (INT8)
  int  uv [320][64];                          // up values (INT8)
  int  hexp [320][64];                        // expected h
  logic signed [15:0] tbl_ref [256];
  int  m0_v, sh_v;

  function automatic int ref_h(input int g, input int u);
    longint p, r;
    p = longint'(tbl_ref[g + 127]) * longint'(u);
    r = (p * longint'(m0_v) + (64'sd1 << (sh_v - 1))) >>> sh_v;
    if (r > 32767) r = 32767;
    if (r < -32767) r = -32767;
    return int'(r);
  endfunction

  task automatic load_table();
    for (int i = 0; i < 255; i++) begin
      @(posedge clk); #1;
      sw_tab_data = {8'b0, i[7:0], tbl_ref[i]};
      sw_tab_we = 1;
    end
    @(posedge clk); #1;
    sw_tab_we = 0;
  endtask

  // Memory image: A = [gi | ui] per row (K = 2 * NOUT), N = 2 * NOUT interleaved identity tiles
  task automatic load_operands(input int M, input int NOUT);
    int subs, K, nt;
    subs = (M + 15) / 16; K = 2 * NOUT; nt = K / 16;
    for (int s = 0; s < subs; s++)
      for (int k = 0; k < K; k++)
        for (int i = 0; i < 16; i++) begin
          int row, v;
          row = s * 16 + i;
          if (row >= M)      v = $urandom_range(255, 0);
          else if (k < NOUT) v = gv[row][k];
          else               v = uv[row][k - NOUT];
          poke(A_BASE + (s * K + k) * 16 + i, v[7:0]);
        end
    for (int t = 0; t < nt; t++) begin
      int tb, src0;
      tb = W_BASE + t * (256 + 16 * K);
      src0 = (t % 2) * NOUT + (t / 2) * 16;        // gate tile j reads column 16 j, up tile j column NOUT + 16 j
      for (int j = 0; j < 16; j++) begin
        for (int b = 0; b < 16; b++) poke(tb + 16 * j + b, 8'h00);
        poke(tb + 16 * j + 7, 8'h40);              // M0 = 2^30
        poke(tb + 16 * j + 8, 8'd30);              // shift 30: y = x
      end
      for (int k = 0; k < K; k++)
        for (int j = 0; j < 16; j++) poke(tb + 256 + 16 * k + j, (src0 + j == k) ? 8'd1 : 8'd0);
    end
  endtask

  task automatic launch(input int M, input int N, input int K, input int spp, input bit out16);
    @(posedge clk); #1;
    op = {16'b0, 8'(spp), 1'b1, 1'b0, 2'b00, 1'b0, out16, 1'b1, 1'b0};
    dim_m = M; dim_n = N; dim_k = K;
    addr_a = A_BASE; addr_w = W_BASE; addr_out = O_BASE; stride_out = 2 * (N / 2);
    sw_m0 = m0_v; sw_shift = sh_v;
    got_done = 0; got_err = 0; got_code = 0;
    start = 1;
    @(posedge clk); #1;
    start = 0;
  endtask

  task automatic wait_end(input int max_cycles, output int cycles);
    cycles = 0;
    while (!got_done && !got_err && cycles < max_cycles) begin @(posedge clk); cycles++; end
    repeat (6) @(posedge clk);
  endtask

  // run one fused fc1 over the data in gv / uv and compare h with hexp
  task automatic run_case(input string name, input int M, input int NOUT, input int spp);
    int cyc, bad, region;
    load_operands(M, NOUT);
    region = M * NOUT * 2;
    for (int a = 0; a < region + 4096; a++) poke(O_BASE + a, 8'hCC);
    launch(M, 2 * NOUT, 2 * NOUT, spp, 1'b0);
    wait_end(3000000, cyc);
    runs++;
    if (!got_done || got_err) begin
      $display("FAIL %s: done=%b err=%b code=%0d", name, got_done, got_err, got_code);
      errors++;
      return;
    end
    bad = 0;
    for (int r = 0; r < M; r++)
      for (int n = 0; n < NOUT; n++) begin
        int a, got, exp;
        a   = O_BASE + r * (2 * NOUT) + 2 * n;
        got = int'(signed'({peek(a + 1), peek(a)}));
        exp = hexp[r][n];
        if (got !== exp) begin
          if (bad < 5) $display("FAIL %s [%0d][%0d]: got %0d expected %0d", name, r, n, got, exp);
          bad++;
        end
      end
    for (int a = region; a < region + 4096; a++)
      if (peek(O_BASE + a) !== 8'hCC) begin
        if (bad < 5) $display("FAIL %s: byte %0d past the output was written", name, a - region);
        bad++; break;
      end
    if (bytes_wr !== region) begin $display("FAIL %s: BYTES_WR %0d expected %0d", name, bytes_wr, region); bad++; end
    if (bad != 0) errors++;
    else $display("  %s: %0d values match, %0d cycles", name, M * NOUT, cyc);
  endtask

  task automatic expect_error(input string name, input int M, input int N, input int spp, input bit out16);
    int cyc;
    load_operands(16, 32);
    @(posedge clk); #1;
    m0_v = 32'h4000_0000; sh_v = 30;
    launch(M, N, 64, spp, out16);
    wait_end(100000, cyc);
    runs++;
    if (!got_err || got_code !== 8'd2) begin $display("FAIL %s: expected error 2, got err=%b code=%0d done=%b", name, got_err, got_code, got_done); errors++; end
  endtask

  // ---- real blocks --------------------------------------------------------------------------------------------
  logic [7:0]  g_m [1024];
  logic [7:0]  u_m [1024];
  logic [15:0] h_m [1024];
  logic [15:0] t_m [256];

  task automatic run_real(input string name, input int spp, input int M);
    int R, C, fd, rc;
    int unsigned mm, ss;
    string d;
    d = {"vectors/", name, "/"};
    fd = $fopen({d, "params.txt"}, "r");
    if (fd == 0) begin $display("FAIL: cannot open %sparams.txt (run tools/dump_swiglu.py)", d); errors++; return; end
    rc = $fscanf(fd, "%d %d %d %d", R, C, mm, ss);
    $fclose(fd);
    $readmemh({d, "gi.hex"}, g_m);
    $readmemh({d, "ui.hex"}, u_m);
    $readmemh({d, "h.hex"}, h_m);
    $readmemh({d, "silu.hex"}, t_m);
    for (int i = 0; i < 255; i++) tbl_ref[i] = t_m[i];
    m0_v = mm; sh_v = ss;
    load_table();
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++) begin
        gv[r][c]   = int'(signed'(g_m[r * C + c]));
        uv[r][c]   = int'(signed'(u_m[r * C + c]));
        hexp[r][c] = int'(signed'(h_m[r * C + c]));
      end
    run_case({"real ", name, " ", $sformatf("%0dx%0d spp=%0d", M, C, spp)}, M, C, spp);
  endtask

  task automatic run_synth(input string name, input int M, input int NOUT, input int spp, input int sh_lo, input int sh_hi);
    for (int i = 0; i < 255; i++) tbl_ref[i] = 16'($urandom_range(65535, 0));
    m0_v = 32'h4000_0000 + $urandom_range(32'h3fff_ffff, 0);
    sh_v = $urandom_range(sh_hi, sh_lo);
    load_table();
    for (int r = 0; r < M; r++)
      for (int c = 0; c < NOUT; c++) begin
        gv[r][c] = $urandom_range(254, 0) - 127;
        uv[r][c] = $urandom_range(254, 0) - 127;
        if ($urandom_range(15, 0) == 0) begin gv[r][c] = 127 - 254 * $urandom_range(1, 0); uv[r][c] = 127 - 254 * $urandom_range(1, 0); end
      end
    for (int r = 0; r < M; r++)
      for (int c = 0; c < NOUT; c++) hexp[r][c] = ref_h(gv[r][c], uv[r][c]);
    run_case({"synthetic ", name, $sformatf(" %0dx%0d spp=%0d shift=%0d", M, NOUT, spp, sh_v)}, M, NOUT, spp);
  endtask

  initial begin
    void'($urandom(32'h5A1B0C0D));
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    run_real("b0_swiglu", 2, 32);
    run_real("b11_swiglu", 1, 32);
    run_real("b0_swiglu", 2, 21);

    run_synth("one panel", 48, 32, 3, 30, 42);
    run_synth("partial subtile", 37, 64, 16, 30, 42);
    run_synth("saturating", 33, 32, 1, 20, 32);
    run_synth("single tile pair", 16, 16, 1, 30, 42);
    stall_en = 1;
    run_synth("write stalls", 100, 48, 4, 30, 42);
    stall_en = 0; block_writes = 0;

    expect_error("16-bit output", 16, 64, 1, 1'b1);
    expect_error("odd tile count", 16, 48, 1, 1'b0);
    expect_error("17 subtiles per panel", 16, 64, 17, 1'b0);

    if (mem_r.bad_access != 0 || mem_w.bad_access != 0) begin
      $display("FAIL: bad memory accesses r=%0d w=%0d", mem_r.bad_access, mem_w.bad_access);
      errors++;
    end
    if (errors == 0) $display("PASS tb_swiglu_fused: %0d runs", runs);
    else             $display("FAIL tb_swiglu_fused: %0d errors in %0d runs", errors, runs);
    $finish;
  end

  initial begin
    #3s;
    $display("FAIL tb_swiglu_fused: global timeout");
    $finish;
  end
endmodule
