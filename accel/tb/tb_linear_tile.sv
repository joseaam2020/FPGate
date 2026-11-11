// End-to-end check of linear_tile (GEMM array + epilogue) with real data from the
// e1_r7 package (tools/dump_vectors.py): each 32x32 slice is split into 16x16 tiles,
// run through the array (two passes for 16-bit inputs) and the epilogue, and compared
// bit by bit with the model's output (y.hex). Two instances run in parallel with
// different LANES (4 and 16) to exercise the drain back-pressure.
`timescale 1ns/1ps

module tb_linear_tile;
  import fpgate_pkg::*;

  localparam int ROWS = PE_ROWS;
  localparam int COLS = PE_COLS;
  localparam int KMAX = 2048;

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic                              in_valid, in_first, in_shift, in_last;
  logic signed [ROWS-1:0][ACT_W-1:0] a_vec;
  logic signed [COLS-1:0][WGT_W-1:0] w_vec;
  logic signed [COLS-1:0][31:0]      bias;
  logic        [COLS-1:0][30:0]      m0;
  logic        [COLS-1:0][5:0]       shift;
  logic                              out16;
  logic                              raw32 = 1'b0;
  logic signed [COLS-1:0][31:0]      rr4, rr16;

  logic                              busy4, busy16;
  logic                              rv4, rv16, rl4, rl16;
  logic signed [COLS-1:0][15:0]      rd4, rd16;

  linear_tile #(.LANES(4))  dut4  (.*, .busy(busy4),  .row_valid(rv4),  .row_data(rd4),  .row_raw(rr4),  .row_last(rl4));
  linear_tile #(.LANES(16)) dut16 (.*, .busy(busy16), .row_valid(rv16), .row_data(rd16), .row_raw(rr16), .row_last(rl16));

  logic signed [15:0] got4  [ROWS][COLS];
  logic signed [15:0] got16 [ROWS][COLS];
  int n4, n16;
  int errors = 0;
  int tiles  = 0;
  int cycles4, cycles16;

  always @(posedge clk) begin
    if (rv4)  begin for (int c = 0; c < COLS; c++) got4[n4][c]   <= rd4[c];  n4  <= n4 + 1;  end
    if (rv16) begin for (int c = 0; c < COLS; c++) got16[n16][c] <= rd16[c]; n16 <= n16 + 1; end
  end

  logic [15:0] x_m [32*KMAX];
  logic [7:0]  w_m [32*KMAX];
  logic [31:0] bias_m [32];
  logic [31:0] m0_m [32];
  logic [7:0]  sh_m [32];
  logic [15:0] y_m [32*32];

  function automatic logic signed [8:0] act_byte(input logic [15:0] v, input int ib, input int pass);
    if (ib == 8)      return 9'(signed'(v[7:0]));
    else if (pass==0) return 9'(signed'(v[15:8]));
    else              return {1'b0, v[7:0]};
  endfunction

  task automatic run_tile(input string name, input int rt, input int ct,
                          input int R, input int C, input int K, input int ib, input int ob);
    int npass = (ib == 8) ? 1 : 2;
    int to, t0;
    logic signed [15:0] e;
    n4 = 0; n16 = 0;
    for (int c = 0; c < COLS; c++) begin
      bias[c]  = bias_m[ct*16 + c];
      m0[c]    = m0_m[ct*16 + c][30:0];
      shift[c] = sh_m[ct*16 + c][5:0];
    end
    out16 = (ob > 8);
    t0 = $time;
    for (int p = 0; p < npass; p++)
      for (int k = 0; k < K; k++) begin
        @(posedge clk); #1;
        in_valid = 1;
        in_first = (k == 0 && p == 0);
        in_shift = (k == 0 && p == 1);
        in_last  = (k == K-1 && p == npass-1);
        for (int i = 0; i < ROWS; i++) a_vec[i] = act_byte(x_m[(rt*16+i)*K + k], ib, p);
        for (int j = 0; j < COLS; j++) w_vec[j] = signed'(w_m[(ct*16+j)*K + k]);
      end
    @(posedge clk); #1;
    in_valid = 0; in_first = 0; in_shift = 0; in_last = 0; a_vec = '0; w_vec = '0;
    to = 0;
    while ((n4 < ROWS || n16 < ROWS || busy4 || busy16) && to < 5000) begin @(posedge clk); to++; end
    repeat (3) @(posedge clk);
    if (n4 != ROWS || n16 != ROWS) begin
      $display("FAIL %s tile(%0d,%0d): rows out L4=%0d L16=%0d", name, rt, ct, n4, n16);
      errors++;
    end
    for (int r = 0; r < ROWS; r++)
      for (int c = 0; c < COLS; c++) begin
        e = (ob > 8) ? y_m[(rt*16+r)*C + ct*16 + c] : 16'(signed'(y_m[(rt*16+r)*C + ct*16 + c][7:0]));
        if (got4[r][c] !== e || got16[r][c] !== e) begin
          if (errors < 10)
            $display("FAIL %s tile(%0d,%0d)[%0d][%0d]: L4=%0d L16=%0d expected %0d",
                     name, rt, ct, r, c, got4[r][c], got16[r][c], e);
          errors++;
        end
      end
    tiles++;
  endtask

  task automatic run_layer(input string name);
    int R, C, K, ib, ob, fd, rc;
    string d;
    d = {"vectors/", name, "/"};
    fd = $fopen({d, "params.txt"}, "r");
    if (fd == 0) begin $display("FAIL: cannot open %sparams.txt (run tools/dump_vectors.py)", d); errors++; return; end
    rc = $fscanf(fd, "%d %d %d %d %d", R, C, K, ib, ob);
    $fclose(fd);
    $readmemh({d, "x.hex"}, x_m);
    $readmemh({d, "w.hex"}, w_m);
    $readmemh({d, "bias.hex"}, bias_m);
    $readmemh({d, "m0.hex"}, m0_m);
    $readmemh({d, "shift.hex"}, sh_m);
    $readmemh({d, "y.hex"}, y_m);
    for (int rt = 0; rt < R/16; rt++)
      for (int ct = 0; ct < C/16; ct++)
        run_tile(name, rt, ct, R, C, K, ib, ob);
    $display("  layer %s: %0d tiles of 16x16, K=%0d, in %0d bit, out %0d bit", name, (R/16)*(C/16), K, ib, ob);
  endtask

  initial begin
    in_valid = 0; in_first = 0; in_shift = 0; in_last = 0; a_vec = '0; w_vec = '0;
    bias = '0; m0 = '0; shift = '0; out16 = 0; n4 = 0; n16 = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    run_layer("b0_q_w");
    run_layer("b0_v_w");
    run_layer("b0_fc1g_w");
    run_layer("b0_fc2_w");
    run_layer("b1_fc2_w");

    if (errors == 0) $display("PASS tb_linear_tile: %0d tiles x 2 configurations", tiles);
    else             $display("FAIL tb_linear_tile: %0d errors in %0d tiles", errors, tiles);
    $finish;
  end

  initial begin
    #200ms;
    $display("FAIL tb_linear_tile: global timeout");
    $finish;
  end
endmodule
