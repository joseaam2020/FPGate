// Output layout writer (OUTFMT 1 / 2 / 3) and RoPE in the epilogue of gemm_engine, bit by bit.
//
// The engine runs identity "projections" (weights = identity, bias 0, M0 = 2^30, shift 30, so y = x) to put exact
// values into the epilogue:
//   * Q, K^T and V of block 0 head 0 (vectors/attn_b0_h0/{q8,k8,v8}.bin, 1025 tokens x 64) go through OUTFMT 1 / 2 / 3;
//     the memory image must equal the layouts the software packs for the attention (full_a.hex for Q, full_wk.hex for K^T,
//     full_wv.hex for V, written by tools/pack_attn_full.py from the same dumps) and the direct formula of every byte.
//   * INT16 values (random, with a spread of magnitudes so that saturated and plain results both occur) go through
//     the RoPE epilogue with the real tables of e1_r7 (vectors/rope, tools/pack_rope.py): the reference is rope_int_range
//     applied with the FULL [1024][64] tables, while the hardware only receives the separable 4 x 512 entry version
//     (sampled from the full table here), so every token tested also proves the separability. Row-major output,
//     OUTFMT 1 and OUTFMT 2, two heads, ROW0 != 0, M not a multiple of 16.
//   * The write path is stalled in bursts during some runs (the transposer must throttle the epilogue, not overflow).
//   * Bad combinations (transposed or RoPE with a 16-bit output) end in error 2.
`timescale 1ns/1ps

module tb_outfmt;
  localparam int A_BEATS = 4096;
  localparam int W_BEATS = 1024;
  localparam int A_BASE  = 32'h00000;
  localparam int W_BASE  = 32'h40000;
  localparam int O_BASE  = 32'h50000;
  localparam int DEPTH   = 1 << 15;          // 1 MiB
  localparam int KP      = 1026;

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

  // bursts of blocked writes while `stall_en` (exercises the throttling of the epilogue)
  bit stall_en = 0;
  initial forever begin
    @(posedge clk);
    if (stall_en && $urandom_range(999, 0) < 4) begin
      block_writes = 1;
      repeat ($urandom_range(900, 60)) @(posedge clk);
      block_writes = 0;
      repeat ($urandom_range(300, 20)) @(posedge clk);
    end
  end

  int errors = 0, runs = 0, sel;
  bit got_done, got_err;
  logic [7:0] got_code;
  always @(posedge clk) begin
    if (hw_done) got_done = 1;
    if (hw_err_valid) begin got_err = 1; got_code = hw_err_code; end
  end

  // ---- memory helpers (both models hold the same image) -----------------------------------------------------
  function automatic logic [7:0] peek(input int a);
    return mem_w.mem[a >> 5][8 * (a & 31) +: 8];
  endfunction
  task automatic poke(input int a, input logic [7:0] b);
    mem_r.mem[a >> 5][8 * (a & 31) +: 8] = b;
    mem_w.mem[a >> 5][8 * (a & 31) +: 8] = b;
  endtask

  // ---- data -------------------------------------------------------------------------------------------------
  int  xr [1040][128];                 // engine input (also the INT16 value after the identity requantisation)
  int  yexp [1040][128];               // expected epilogue output

  logic [7:0]  qb [65600];
  logic [7:0]  kb [65600];
  logic [7:0]  vb [65600];
  logic [15:0] ctab [65536];
  logic [15:0] stab [65536];
  int rope_frac, rope_pre;

  task automatic load_vectors();
    int fd, n, rc;
    fd = $fopen("vectors/attn_b0_h0/q8.bin", "rb"); n = $fread(qb, fd); $fclose(fd);
    if (n != 65600) begin $display("FAIL: q8.bin has %0d bytes", n); errors++; end
    fd = $fopen("vectors/attn_b0_h0/k8.bin", "rb"); n = $fread(kb, fd); $fclose(fd);
    fd = $fopen("vectors/attn_b0_h0/v8.bin", "rb"); n = $fread(vb, fd); $fclose(fd);
    $readmemh("vectors/rope/cos.hex", ctab);
    $readmemh("vectors/rope/sin.hex", stab);
    fd = $fopen("vectors/rope/params.txt", "r");
    rc = $fscanf(fd, "%d %d", rope_frac, rope_pre);
    $fclose(fd);
  endtask

  // Memory image of an identity projection: A (tile-major, one or two planes), W (N/16 tiles, K words each)
  task automatic load_operands(input int M, input int N, input int K, input bit in16);
    int subs, planes, nt;
    subs = (M + 15) / 16; planes = in16 ? 2 : 1; nt = N / 16;
    for (int s = 0; s < subs; s++)
      for (int p = 0; p < planes; p++)
        for (int k = 0; k < K; k++)
          for (int i = 0; i < 16; i++) begin
            int row, v;
            logic [7:0] b;
            row = s * 16 + i;
            v   = (row < M) ? xr[row][k] : $urandom_range(255, 0);
            if (!in16)       b = v[7:0];
            else if (p == 0) b = v[15:8];
            else             b = v[7:0];
            poke(A_BASE + ((s * planes + p) * K + k) * 16 + i, b);
          end
    for (int t = 0; t < nt; t++) begin
      int tb;
      tb = W_BASE + t * (256 + 16 * K);
      for (int j = 0; j < 16; j++) begin
        for (int b = 0; b < 16; b++) poke(tb + 16 * j + b, 8'h00);
        poke(tb + 16 * j + 7, 8'h40);                  // M0 = 2^30
        poke(tb + 16 * j + 8, 8'd30);                  // shift 30: y = x
      end
      for (int k = 0; k < K; k++)
        for (int j = 0; j < 16; j++) poke(tb + 256 + 16 * k + j, (t * 16 + j == k) ? 8'd1 : 8'd0);
    end
  endtask

  task automatic fill_output(input int base, input int bytes);
    for (int a = 0; a < bytes; a++) poke(base + a, 8'hCC);
  endtask

  task automatic load_rope_tables();
    // sampled from the full tables: d < 32 from the token with x = 0 (y = yx), d >= 32 from the token with y = 0 (x = yx)
    for (int sn = 0; sn < 2; sn++)
      for (int half = 0; half < 2; half++)
        for (int yx = 0; yx < 32; yx++)
          for (int pi = 0; pi < 16; pi++) begin
            int u, d;
            logic [15:0] v;
            u = half ? yx : 32 * yx;
            d = 32 * half + 2 * pi;
            v = sn ? stab[u * 64 + d] : ctab[u * 64 + d];
            @(posedge clk); #1;
            rope_tab_data = {5'b0, sn[0], 10'({half[0], yx[4:0], pi[3:0]}), v};
            rope_tab_we = 1;
            @(posedge clk); #1;
            rope_tab_we = 0;
          end
  endtask

  // rope_int_range of the software, with the full tables
  function automatic int rope_ref(input int t, input int d, input int xe, input int xo_pair, input int frac, input int pre);
    longint v, rot, c, s, rnd;
    int sh;
    sh = frac + pre;
    rnd = 64'sd1 <<< (sh - 1);
    if (t == 0) begin
      v = (longint'(xe) + (64'sd1 <<< (pre - 1))) >>> pre;
    end else begin
      c = longint'(signed'(ctab[(t - 1) * 64 + d]));
      s = longint'(signed'(stab[(t - 1) * 64 + d]));
      rot = (d & 1) ? longint'(xo_pair) : -longint'(xo_pair);
      v = (longint'(xe) * c + rot * s + rnd) >>> sh;
    end
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    return int'(v);
  endfunction

  // ---- run -------------------------------------------------------------------------------------------------
  task automatic launch(input int M, input int N, input int K, input bit in16, input int spp, input int fmt, input bit rope,
                        input int stride, input int obase, input int tstride, input int htiles, input int hstride, input int r0);
    @(posedge clk); #1;
    op = {16'b0, 8'(spp), 1'b0, rope, 2'(fmt), 1'b0, 1'b0, 1'b1, in16};
    dim_m = M; dim_n = N; dim_k = K;
    addr_a = A_BASE; addr_w = W_BASE; addr_out = obase; stride_out = stride;
    out_tstride = tstride; head_tiles = htiles; head_stride = hstride; row0 = r0;
    rope_cfg = {18'b0, 6'(rope_pre), 2'b0, 6'(rope_frac + rope_pre)};
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

  // expected byte of the output image
  //   fmt 0: base + r * stride + n            fmt 1: Q blocks               fmt 2: K^T blocks (base already past the header)
  //   fmt 3: V tiles (base already past the header, stride = 16, tstride = 256 + 16 * KP)
  function automatic int out_addr(input int fmt, input int r, input int n, input int base, input int stride, input int hd, input int S);
    int h, d, s, j;
    h = n / hd; d = n % hd; s = r / 16; j = r % 16;
    case (fmt)
      0: return base + r * stride + n;
      1: return base + h * S * hd * 16 + s * hd * 16 + d * 16 + j;
      2: return base + h * S * (hd * 16 + 256) + s * (hd * 16 + 256) + d * 16 + j;
      default: return base + (n / 16) * (256 + 16 * KP) + r * 16 + (n % 16);
    endcase
  endfunction

  // run one projection and compare the whole image; `name` labels the failures. M rows, N channels (heads of hd)
  task automatic run_case(input string name, input int M, input int N, input int K, input bit in16, input int spp,
                          input int fmt, input bit rope, input int r0, input int hd);
    int S, stride, obase, tstride, htiles, hstride, cyc, bad, region, nt, exp_wr;
    S = (M + 15) / 16; nt = N / 16;
    stride = N; obase = O_BASE; tstride = 0; htiles = 0; hstride = 0;
    case (fmt)
      1: begin stride = hd; tstride = 256; htiles = hd / 16; hstride = S * hd * 16; region = (N / hd) * S * hd * 16; end
      2: begin stride = hd + 16; obase = O_BASE + 256; tstride = 256; htiles = hd / 16; hstride = S * (hd * 16 + 256);
               region = (N / hd) * S * (hd * 16 + 256) + 256; end
      3: begin stride = 16; obase = O_BASE + 256; tstride = 256 + 16 * KP; region = nt * (256 + 16 * KP) + 256; end
      default: begin
        stride = N; region = M * N + 64;
        if (rope && (N > hd)) begin htiles = hd / 16; hstride = hd; end    // heads side by side: same as the natural walk
      end
    endcase
    load_operands(M, N, K, in16);
    fill_output(O_BASE, region + 64);
    launch(M, N, K, in16, spp, fmt, rope, stride, obase, tstride, htiles, hstride, r0);
    wait_end(600000, cyc);
    runs++;
    if (!got_done || got_err) begin
      $display("FAIL %s: done=%b err=%b code=%0d", name, got_done, got_err, got_code);
      errors++;
      return;
    end
    bad = 0;
    for (int r = 0; r < M; r++)
      for (int n = 0; n < N; n++) begin
        int a, got, exp;
        a = out_addr(fmt, r, n, obase, stride, hd, S);
        got = int'(signed'(peek(a)));
        exp = yexp[r][n];
        if (got !== exp) begin
          if (bad < 5) $display("FAIL %s [%0d][%0d] @%0d: got %0d expected %0d", name, r, n, a, got, exp);
          bad++;
        end
      end
    if (fmt == 1 || fmt == 2)
      for (int r = M; r < 16 * S; r++)           // padding rows of the last subtile are written as zero
        for (int n = 0; n < N; n++) begin
          int a;
          a = out_addr(fmt, r, n, obase, stride, hd, S);
          if (peek(a) !== 8'h00) begin if (bad < 5) $display("FAIL %s: padding row %0d channel %0d is not zero", name, r, n); bad++; end
        end
    if (fmt == 2)
      for (int s = 0; s < S; s++)                // the 256-byte header of every tile is not touched
        for (int b = 0; b < 256; b++)
          if (peek(O_BASE + s * (hd * 16 + 256) + b) !== 8'hCC) begin
            if (bad < 5) $display("FAIL %s: header of tile %0d was written", name, s);
            bad++; break;
          end
    if (fmt == 3)
      for (int t = 0; t < nt; t++) begin
        for (int b = 0; b < 256; b++)
          if (peek(O_BASE + t * (256 + 16 * KP) + b) !== 8'hCC) begin
            if (bad < 5) $display("FAIL %s: header of V tile %0d was written", name, t);
            bad++; break;
          end
      end
    exp_wr = (fmt == 1 || fmt == 2) ? nt * S * 8 * 32 : M * nt * 16;
    if (bytes_wr !== exp_wr) begin $display("FAIL %s: BYTES_WR %0d expected %0d", name, bytes_wr, exp_wr); bad++; end
    if (bad != 0) errors++;
    else $display("  %s: %0d values match, %0d cycles", name, M * N, cyc);
  endtask

  task automatic expect_error(input string name, input int fmt, input bit rope, input bit out16);
    int cyc;
    load_operands(16, 16, 16, 0);
    @(posedge clk); #1;
    op = {16'b0, 8'd2, 1'b0, rope, 2'(fmt), 1'b0, out16, 1'b1, 1'b0};
    dim_m = 16; dim_n = 16; dim_k = 16;
    addr_a = A_BASE; addr_w = W_BASE; addr_out = O_BASE; stride_out = 16;
    out_tstride = 0; head_tiles = 0; head_stride = 0; row0 = 0;
    got_done = 0; got_err = 0; got_code = 0;
    start = 1;
    @(posedge clk); #1;
    start = 0;
    wait_end(100000, cyc);
    runs++;
    if (!got_err || got_code !== 8'd2) begin $display("FAIL %s: expected error 2, got err=%b code=%0d done=%b", name, got_err, got_code, got_done); errors++; end
  endtask

  // ---- reference builders -------------------------------------------------------------------------------------
  task automatic set_identity(input int M, input int N, input int K, input logic [7:0] src [65600], input int hd);
    for (int r = 0; r < M; r++)
      for (int n = 0; n < N; n++) begin
        xr[r][n]   = int'(signed'(src[r * hd + (n % hd)]));
        yexp[r][n] = xr[r][n];
      end
  endtask

  // INT16 inputs with a spread of magnitudes; expected result = RoPE of the pair (token = r0 + row)
  task automatic set_rope_data(input int M, input int N, input int r0, input int hd);
    for (int r = 0; r < M; r++)
      for (int n = 0; n < N; n++) begin
        int m;
        m = $urandom_range(32767, 0) >> $urandom_range(12, 0);
        xr[r][n] = $urandom_range(1, 0) ? -m : m;
        if ($urandom_range(40, 0) == 0) xr[r][n] = $urandom_range(1, 0) ? 32767 : -32767;
      end
    for (int r = 0; r < M; r++)
      for (int n = 0; n < N; n++) begin
        int d, pair;
        d = n % hd;
        pair = (d & 1) ? xr[r][n - 1] : xr[r][n + 1];
        yexp[r][n] = rope_ref(r0 + r, d, xr[r][n], pair, rope_frac, rope_pre);
      end
  endtask

  // cross-checks against the pack_attn_full.py images
  logic [255:0] hq [2 * 64 / 2 * 2];
  logic [255:0] hk [65 * 40];
  logic [255:0] hv [4 * 521];

  task automatic cross_q(input int base);
    int bad;
    logic [255:0] want, got;
    $readmemh("vectors/attn_b0_h0/full_a.hex", hq);
    bad = 0;
    for (int s = 0; s < 2; s++)                // 32 rows = 2 subtiles; the tile-major image of Q: 64 words per subtile
      for (int w = 0; w < 64; w += 2) begin
        want = hq[s * 32 + w / 2];
        for (int b = 0; b < 32; b++) got[8 * b +: 8] = peek(base + s * 64 * 16 + w * 16 + b);
        if (got !== want) begin bad++; if (bad < 3) $display("FAIL cross Q: subtile %0d word %0d", s, w); end
      end
    if (bad) errors++; else $display("  Q image equals full_a.hex (2 subtiles)");
  endtask

  task automatic cross_k(input int base);
    int bad;
    logic [255:0] got;
    $readmemh("vectors/attn_b0_h0/full_wk.hex", hk);
    bad = 0;
    for (int t = 0; t < 65; t++)
      for (int i = 0; i < 32; i++) begin
        for (int b = 0; b < 32; b++) got[8 * b +: 8] = peek(base + t * 1280 + i * 32 + b);
        if (got !== hk[t * 40 + 8 + i]) begin bad++; if (bad < 3) $display("FAIL cross K^T: tile %0d beat %0d", t, i); end
      end
    if (bad) errors++; else $display("  K^T image equals full_wk.hex (65 tiles)");
  endtask

  task automatic cross_v(input int base);
    int bad;
    logic [255:0] got;
    $readmemh("vectors/attn_b0_h0/full_wv.hex", hv);
    bad = 0;
    for (int t = 0; t < 4; t++)
      for (int i = 0; i < 512; i++) begin          // words 0 .. 1023 (word 1024 is checked by the formula, 1025 is the pad)
        for (int b = 0; b < 32; b++) got[8 * b +: 8] = peek(base + t * (256 + 16 * KP) + i * 32 + b);
        if (got !== hv[t * 521 + 8 + i]) begin bad++; if (bad < 3) $display("FAIL cross V: tile %0d beat %0d", t, i); end
      end
    if (bad) errors++; else $display("  V image equals full_wv.hex (4 tiles, 1024 words each)");
  endtask

  initial begin
    void'($urandom(32'h41F0E0A1));
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);
    load_vectors();

    // plusarg +sel=N runs one group only (default: all)
    if (!$value$plusargs("sel=%d", sel)) sel = -1;

    if (sel < 0 || sel == 1) begin
    // two heads, M not a multiple of 16, one subtile per panel
    set_identity(40, 128, 128, qb, 64);
    run_case("Q fmt1 40x128 two heads spp=1", 40, 128, 128, 0, 1, 1, 0, 0, 64);
    run_case("K^T fmt2 40x128 two heads spp=3", 40, 128, 128, 0, 3, 2, 0, 0, 64);
    run_case("fmt0 row-major, head jump disabled", 17, 64, 64, 0, 2, 0, 0, 0, 64);
    set_identity(40, 64, 64, vb, 64);
    run_case("V fmt3 40x64", 40, 64, 64, 0, 2, 3, 0, 0, 64);
    end

    if (sel < 0 || sel == 0) begin
    // ---- real Q, K, V of one head ----------------------------------------------------------------------------
    set_identity(1025, 64, 64, qb, 64);
    run_case("Q  fmt1 1025x64 (real)", 1025, 64, 64, 0, 16, 1, 0, 0, 64);
    cross_q(O_BASE);
    set_identity(1025, 64, 64, kb, 64);
    run_case("K^T fmt2 1025x64 (real)", 1025, 64, 64, 0, 16, 2, 0, 0, 64);
    cross_k(O_BASE + 256);
    set_identity(1025, 64, 64, vb, 64);
    run_case("V  fmt3 1025x64 (real)", 1025, 64, 64, 0, 16, 3, 0, 0, 64);
    cross_v(O_BASE + 256);
    end

    if (sel < 0 || sel == 2) begin
    // writes stalled in bursts: the transposer throttles the epilogue
    stall_en = 1;
    set_identity(200, 64, 64, qb, 64);
    run_case("Q fmt1 200x64 write stalls", 200, 64, 64, 0, 4, 1, 0, 0, 64);
    stall_en = 0; block_writes = 0;
    end

    if (sel < 0 || sel == 3) begin
    // ---- RoPE -------------------------------------------------------------------------------------------------
    load_rope_tables();
    set_rope_data(1025, 64, 0, 64);
    run_case("RoPE row-major 1025x64 (tokens 0..1024, in16)", 1025, 64, 64, 1, 8, 0, 1, 0, 64);
    set_rope_data(37, 128, 700, 64);
    run_case("RoPE row-major 37x128 two heads, ROW0=700", 37, 128, 128, 1, 2, 0, 1, 700, 64);
    set_rope_data(53, 128, 1, 64);
    run_case("RoPE + Q fmt1 53x128, ROW0=1", 53, 128, 128, 1, 2, 1, 1, 1, 64);
    set_rope_data(40, 128, 0, 64);
    run_case("RoPE + K^T fmt2 40x128, ROW0=0 (class token)", 40, 128, 128, 1, 3, 2, 1, 0, 64);
    stall_en = 1;
    set_rope_data(150, 64, 870, 64);
    run_case("RoPE + K^T fmt2 150x64 write stalls", 150, 64, 64, 1, 4, 2, 1, 870, 64);
    stall_en = 0; block_writes = 0;
    end

    if (sel < 0 || sel == 4) begin
    // ---- bad combinations -------------------------------------------------------------------------------------
    expect_error("transposed with 16-bit output", 1, 0, 1);
    expect_error("RoPE with 16-bit output", 0, 1, 1);
    end

    if (mem_r.bad_access != 0 || mem_w.bad_access != 0) begin
      $display("FAIL: bad memory accesses r=%0d w=%0d", mem_r.bad_access, mem_w.bad_access);
      errors++;
    end
    if (errors == 0) $display("PASS tb_outfmt: %0d runs", runs);
    else             $display("FAIL tb_outfmt: %0d errors in %0d runs", errors, runs);
    $finish;
  end

  initial begin
    #3s;
    $display("FAIL tb_outfmt: global timeout");
    $finish;
  end
endmodule
