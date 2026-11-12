// Self-checking testbench for gemm_array (and therefore pe).
// Modes: 0 = INT8 activations, 1 = signed INT16 (fc2), 2 = unsigned UINT16 (P in P*V).
// 16-bit modes run as two passes (high byte, low byte) with the shifting accumulator.
// The expected value is computed directly with 64-bit math and compared modulo 2^32.
`timescale 1ns/1ps

module tb_gemm_array;
  import fpgate_pkg::*;

  localparam int ROWS = 16;
  localparam int COLS = 16;
  localparam int KMAX = 2048;
  localparam int SEED = 32'h00F96A7E;

  logic clk = 0;
  logic rst_n = 0;
  always #3.333 clk = ~clk;

  logic                              in_valid, in_first, in_shift, in_last;
  logic signed [ROWS-1:0][ACT_W-1:0] a_vec;
  logic signed [COLS-1:0][WGT_W-1:0] w_vec;
  logic                              busy, drain_valid, drain_last, last_ok, drain_tag;
  logic                              in_tag = 1'b0;
  logic                              drain_ready = 1'b1;
  logic [$clog2(ROWS)-1:0]           drain_row;
  logic signed [COLS-1:0][ACC_W-1:0] drain_data;

  gemm_array #(.ROWS(ROWS), .COLS(COLS)) dut (.*);

  int x [ROWS][KMAX];   // activations (full value, 8 or 16 bit)
  int w [KMAX][COLS];   // weights (INT8)
  logic signed [ACC_W-1:0] got [ROWS][COLS];
  int rows_seen;
  int errors = 0;
  int tiles  = 0;

  always @(posedge clk) begin
    if (drain_valid) begin
      for (int j = 0; j < COLS; j++) got[drain_row][j] <= drain_data[j];
      rows_seen <= rows_seen + 1;
    end
  end

  function automatic logic signed [8:0] act_byte(input int v, input int mode, input int pass);
    case (mode)
      0:       return 9'(v);                       // INT8, sign-extended
      1:       return pass == 0 ? 9'(v >>> 8)      // signed high byte
                                : {1'b0, 8'(v)};   // unsigned low byte
      default: return pass == 0 ? {1'b0, 8'(v >> 8)}
                                : {1'b0, 8'(v)};
    endcase
  endfunction

  task automatic run_tile(input int mode, input int K, input string name);
    int npass = (mode == 0) ? 1 : 2;
    longint exp_v;
    logic signed [ACC_W-1:0] exp32;
    int to;
    rows_seen = 0;
    for (int p = 0; p < npass; p++) begin
      for (int k = 0; k < K; k++) begin
        @(posedge clk); #1;
        in_valid = 1;
        in_first = (k == 0 && p == 0);
        in_shift = (k == 0 && p == 1);
        in_last  = (k == K-1 && p == npass-1);
        for (int i = 0; i < ROWS; i++) a_vec[i] = act_byte(x[i][k], mode, p);
        for (int j = 0; j < COLS; j++) w_vec[j] = 8'(w[k][j]);
      end
    end
    @(posedge clk); #1;
    in_valid = 0; in_first = 0; in_shift = 0; in_last = 0;
    a_vec = '0; w_vec = '0;
    to = 0;
    while (rows_seen < ROWS && to < 1000) begin @(posedge clk); to++; end
    @(posedge clk); #1;
    if (rows_seen != ROWS || busy) begin
      $display("FAIL %s K=%0d: drained %0d rows, busy=%b", name, K, rows_seen, busy);
      errors++;
    end
    for (int i = 0; i < ROWS; i++) begin
      for (int j = 0; j < COLS; j++) begin
        exp_v = 0;
        for (int k = 0; k < K; k++) exp_v += longint'(x[i][k]) * longint'(w[k][j]);
        exp32 = exp_v[ACC_W-1:0];
        if (got[i][j] !== exp32) begin
          if (errors < 10)
            $display("FAIL %s K=%0d [%0d][%0d]: got %0d, expected %0d", name, K, i, j, got[i][j], exp32);
          errors++;
        end
      end
    end
    tiles++;
  endtask

  // value generators ---------------------------------------------------------
  function automatic int rnd_act(input int mode);
    case (mode)
      0:       return int'($urandom_range(255, 0)) - 128;
      1:       return int'($urandom_range(65535, 0)) - 32768;
      default: return int'($urandom_range(65535, 0));
    endcase
  endfunction

  task automatic fill_random(input int mode, input int K);
    for (int i = 0; i < ROWS; i++) for (int k = 0; k < K; k++) x[i][k] = rnd_act(mode);
    for (int k = 0; k < K; k++) for (int j = 0; j < COLS; j++) w[k][j] = int'($urandom_range(255, 0)) - 128;
  endtask

  task automatic fill_const(input int xv, input int wv, input int K);
    for (int i = 0; i < ROWS; i++) for (int k = 0; k < K; k++) x[i][k] = xv;
    for (int k = 0; k < K; k++) for (int j = 0; j < COLS; j++) w[k][j] = wv;
  endtask

  int Ks [] = '{1, 2, 3, 16, 17, 48, 100, 768};

  initial begin
    void'($urandom(SEED));
    in_valid = 0; in_first = 0; in_shift = 0; in_last = 0;
    a_vec = '0; w_vec = '0;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    // random data, every mode and a spread of K (including non-multiples of 16)
    for (int m = 0; m < 3; m++)
      foreach (Ks[n]) begin
        fill_random(m, Ks[n]);
        run_tile(m, Ks[n], $sformatf("random mode%0d", m));
      end

    // corners
    fill_const(-128, -128, 64);       run_tile(0, 64, "int8 all -128*-128");
    fill_const(127, -128, 64);        run_tile(0, 64, "int8 127*-128");
    fill_const(0, 0, 33);             run_tile(0, 33, "int8 zeros");
    fill_const(-32768, -128, 64);     run_tile(1, 64, "int16 -32768*-128");
    fill_const(32767, -128, 64);      run_tile(1, 64, "int16 32767*-128");
    fill_const(-1, 127, 64);          run_tile(1, 64, "int16 -1*127");
    fill_const(65535, -128, 64);      run_tile(2, 64, "uint16 65535*-128");
    fill_const(65535, 127, 2048);     run_tile(2, 2048, "uint16 65535*127 K=2048 (wraps mod 2^32)");
    fill_const(-32768, -128, 2048);   run_tile(1, 2048, "int16 -32768*-128 K=2048");

    // long random tile
    fill_random(1, 1232);             run_tile(1, 1232, "int16 K=1232");

    if (errors == 0) $display("PASS tb_gemm_array: %0d tiles", tiles);
    else             $display("FAIL tb_gemm_array: %0d errors in %0d tiles", errors, tiles);
    $finish;
  end

  initial begin
    #50ms;
    $display("FAIL tb_gemm_array: global timeout");
    $finish;
  end
endmodule
