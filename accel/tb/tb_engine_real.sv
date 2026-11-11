// Real data through the whole engine: slices of the e1_r7 package (dumped by
// tools/dump_vectors.py, packed into the DDR3 formats by tools/pack_layer.py) are loaded
// into the simulated DDR3, an operation is started with the CSR-style parameters, and
// every output is compared bit by bit with the model's own result (y.hex).
// Each layer runs twice: one panel (2 subtiles) and one subtile per panel.
`timescale 1ns/1ps

module tb_engine_real;
  localparam int A_BEATS = 4096;            // 128 KiB: 2 subtiles of 16-bit K=2048
  localparam int W_BEATS = 1024;            // K up to 2048
  localparam int A_BASE  = 32'h00000;
  localparam int W_BASE  = 32'h40000;
  localparam int O_BASE  = 32'h80000;
  localparam int DEPTH   = 1 << 15;

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic         start, soft_reset;
  logic [31:0]  op, dim_m, dim_n, dim_k, addr_a, addr_w, addr_out, stride_out;
  logic         hw_done, hw_err_valid;
  logic [7:0]   hw_err_code;
  logic [31:0]  bytes_rd, bytes_wr;

  logic [31:0]  rd_address, wr_address, wr_byteenable;
  logic         rd_read, rd_waitrequest, rd_readdatavalid, wr_write, wr_waitrequest;
  logic [7:0]   rd_burstcount, wr_burstcount;
  logic [255:0] rd_readdata, wr_writedata;

  gemm_engine #(.A_BEATS(A_BEATS), .W_BEATS(W_BEATS), .LANES(4)) dut (.*);

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

  int errors = 0, runs = 0;
  // gap between consecutive subtile starts inside a tile (cycles); steady-state overhead = gap - planes*K
  int last_fd = 0, min_gap = 1 << 30;
  always @(posedge clk) if (dut.fd_start) begin
    if (last_fd != 0 && ($time - last_fd) / 6.666 < min_gap) min_gap = int'(($time - last_fd) / 6.666);
    last_fd = $time;
  end
  string names [5] = '{"b0_q_w", "b0_v_w", "b0_fc1g_w", "b0_fc2_w", "b1_fc2_w"};
  bit got_done, got_err;
  logic [7:0] got_code;
  always @(posedge clk) begin
    if (hw_done) got_done = 1;
    if (hw_err_valid) begin got_err = 1; got_code = hw_err_code; end
  end

  logic [255:0] img_a [A_BEATS];
  logic [255:0] img_w [4 * (8 + W_BEATS)];
  logic [15:0]  y_m [32 * 32];

  task automatic run_layer(input string name, input int spp);
    int R, C, K, ib, ob, fd, rc, stride, nbeats_a, nbeats_w, cyc, bad, subs, nt, planes, exp_rd, exp_wr;
    int panel_sub, left;
    string d;
    d = {"vectors/", name, "/"};
    fd = $fopen({d, "params.txt"}, "r");
    if (fd == 0) begin $display("FAIL: cannot open %sparams.txt (run tools/dump_vectors.py)", d); errors++; return; end
    rc = $fscanf(fd, "%d %d %d %d %d", R, C, K, ib, ob);
    $fclose(fd);
    planes = (ib > 8) ? 2 : 1; subs = (R + 15) / 16; nt = C / 16;
    nbeats_a = subs * planes * K / 2;
    nbeats_w = nt * (8 + K / 2);
    $readmemh({d, "mem_a.hex"}, img_a);
    $readmemh({d, "mem_w.hex"}, img_w);
    $readmemh({d, "y.hex"}, y_m);
    stride = (ob > 8) ? 96 : 64;
    for (int i = 0; i < nbeats_a; i++) begin mem_r.mem[A_BASE / 32 + i] = img_a[i]; mem_w.mem[A_BASE / 32 + i] = img_a[i]; end
    for (int i = 0; i < nbeats_w; i++) begin mem_r.mem[W_BASE / 32 + i] = img_w[i]; mem_w.mem[W_BASE / 32 + i] = img_w[i]; end
    for (int i = 0; i < (R * stride) / 32 + 2; i++) begin
      mem_r.mem[O_BASE / 32 + i] = {32{8'hCC}}; mem_w.mem[O_BASE / 32 + i] = {32{8'hCC}};
    end

    @(posedge clk); #1;
    op = {16'b0, 8'(spp), 5'b0, (ob > 8), 1'b1, (ib > 8)};
    dim_m = R; dim_n = C; dim_k = K;
    addr_a = A_BASE; addr_w = W_BASE; addr_out = O_BASE; stride_out = stride;
    got_done = 0; got_err = 0; got_code = 0;
    start = 1;
    @(posedge clk); #1;
    start = 0;
    cyc = 0;
    while (!got_done && !got_err && cyc < 3000000) begin @(posedge clk); cyc++; end
    repeat (4) @(posedge clk);
    runs++;
    if (!got_done || got_err) begin
      $display("FAIL %s spp=%0d: done=%b err=%b code=%0d", name, spp, got_done, got_err, got_code);
      errors++;
      return;
    end

    bad = 0;
    for (int r = 0; r < R; r++)
      for (int n = 0; n < C; n++) begin
        logic [255:0] beatv;
        int off, got, exp;
        off  = O_BASE + r * stride + n * ((ob > 8) ? 2 : 1);
        beatv = mem_w.mem[off / 32];
        if (ob > 8) begin
          got = int'(signed'(beatv[8 * (off % 32) +: 16]));
          exp = int'(signed'(y_m[r * C + n]));
        end else begin
          got = int'(signed'(beatv[8 * (off % 32) +: 8]));
          exp = int'(signed'(y_m[r * C + n][7:0]));
        end
        if (got !== exp) begin
          if (bad < 5) $display("FAIL %s spp=%0d [%0d][%0d]: got %0d expected %0d", name, spp, r, n, got, exp);
          bad++;
        end
      end

    exp_rd = 0; left = subs;
    while (left > 0) begin
      panel_sub = (left < spp) ? left : spp;
      exp_rd += panel_sub * planes * (K / 2) + nt * (8 + K / 2);
      left -= panel_sub;
    end
    exp_rd *= 32;
    exp_wr = R * nt * ((ob > 8) ? 32 : 16);
    if (bytes_rd !== exp_rd || bytes_wr !== exp_wr) begin
      $display("FAIL %s spp=%0d: bytes rd/wr %0d/%0d expected %0d/%0d", name, spp, bytes_rd, bytes_wr, exp_rd, exp_wr);
      bad++;
    end
    if (bad != 0) errors++;
    else $display("  %s spp=%0d: %0d outputs match, %0d cycles; subtile gap %0d cycles for %0d feed cycles (overhead %0d)", name, spp, R * C, cyc, min_gap, planes * K, min_gap - planes * K);
    min_gap = 1 << 30; last_fd = 0;
  endtask

  initial begin
    void'($urandom(32'h0E0A1));
    start = 0; soft_reset = 0; op = 0; dim_m = 0; dim_n = 0; dim_k = 0;
    addr_a = 0; addr_w = 0; addr_out = 0; stride_out = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    foreach (names[i]) begin
      run_layer(names[i], 2);
      run_layer(names[i], 1);
    end

    if (errors == 0) $display("PASS tb_engine_real: %0d runs", runs);
    else             $display("FAIL tb_engine_real: %0d errors in %0d runs", errors, runs);
    $finish;
  end


  initial begin
    #3s;
    $display("FAIL tb_engine_real: global timeout");
    $finish;
  end
endmodule
