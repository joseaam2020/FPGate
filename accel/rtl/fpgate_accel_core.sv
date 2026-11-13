// Accelerator core: two register blocks behind one Avalon-MM slave, the GEMM engine and the attention
// softmax unit, each with its own pair of Avalon-MM masters towards the HPS SDRAM port (Platform
// Designer arbitrates the four onto it).
//
//   register file, byte offsets:  0x000..0x0FF  GEMM engine   (csr.sv)
//                                 0x100..0x1FF  softmax unit  (csr_sm.sv)
//
// (fpgate_accel_top, the H0 placeholder, stays untouched because the H0 board design still uses it.)
module fpgate_accel_core #(
  parameter int A_BEATS  = 8192,
  parameter int W_BEATS  = 1024,
  parameter int LANES    = 4,
  parameter int SM_LANES = 4,
  parameter logic [31:0] ADDR_LO = 32'h3000_0000,
  parameter logic [31:0] ADDR_HI = 32'h4000_0000
)(
  input  logic         clk,
  input  logic         rst_n,
  // register slave
  input  logic [6:0]   avs_address,
  input  logic         avs_write,
  input  logic [31:0]  avs_writedata,
  input  logic [3:0]   avs_byteenable,
  input  logic         avs_read,
  output logic [31:0]  avs_readdata,
  output logic         irq,
  // GEMM engine: read master
  output logic [31:0]  rd_address,
  output logic         rd_read,
  output logic [7:0]   rd_burstcount,
  input  logic         rd_waitrequest,
  input  logic [255:0] rd_readdata,
  input  logic         rd_readdatavalid,
  // GEMM engine: write master
  output logic [31:0]  wr_address,
  output logic         wr_write,
  output logic [255:0] wr_writedata,
  output logic [31:0]  wr_byteenable,
  output logic [7:0]   wr_burstcount,
  input  logic         wr_waitrequest,
  // softmax unit: read master
  output logic [31:0]  rd2_address,
  output logic         rd2_read,
  output logic [7:0]   rd2_burstcount,
  input  logic         rd2_waitrequest,
  input  logic [255:0] rd2_readdata,
  input  logic         rd2_readdatavalid,
  // softmax unit: write master
  output logic [31:0]  wr2_address,
  output logic         wr2_write,
  output logic [255:0] wr2_writedata,
  output logic [31:0]  wr2_byteenable,
  output logic [7:0]   wr2_burstcount,
  input  logic         wr2_waitrequest
);
  // ---- address decode ------------------------------------------------------------------------------
  wire sel_sm = avs_address[6];
  logic sel_sm_q;
  always_ff @(posedge clk) if (avs_read) sel_sm_q <= sel_sm;

  // ---- GEMM engine ---------------------------------------------------------------------------------
  logic        start, soft_reset, hw_done, hw_err_valid;
  logic [7:0]  hw_err_code;
  logic [31:0] op, dim_m, dim_n, dim_k, addr_a, addr_w, addr_out, addr_bias, addr_scale, stride_a, stride_out;
  logic [31:0] bytes_rd, bytes_wr;
  logic [31:0] eng_readdata;

  csr u_csr (
    .clk, .rst_n,
    .avs_address(avs_address[5:0]), .avs_write(avs_write && !sel_sm), .avs_writedata, .avs_byteenable,
    .avs_read(avs_read && !sel_sm), .avs_readdata(eng_readdata), .irq,
    .start, .soft_reset, .op, .dim_m, .dim_n, .dim_k,
    .addr_a, .addr_w, .addr_out, .addr_bias, .addr_scale, .stride_a, .stride_out,
    .hw_done, .hw_err_valid, .hw_err_code, .bytes_rd, .bytes_wr
  );

  gemm_engine #(.A_BEATS(A_BEATS), .W_BEATS(W_BEATS), .LANES(LANES), .ADDR_LO(ADDR_LO), .ADDR_HI(ADDR_HI)) u_engine (
    .clk, .rst_n, .start, .soft_reset,
    .op, .dim_m, .dim_n, .dim_k, .addr_a, .addr_w, .addr_out, .stride_out,
    .hw_done, .hw_err_valid, .hw_err_code, .bytes_rd, .bytes_wr,
    .rd_address, .rd_read, .rd_burstcount, .rd_waitrequest, .rd_readdata, .rd_readdatavalid,
    .wr_address, .wr_write, .wr_writedata, .wr_byteenable, .wr_burstcount, .wr_waitrequest
  );

  // ---- softmax unit --------------------------------------------------------------------------------
  logic        sm_start, sm_soft_reset, sm_done, sm_err_valid;
  logic [7:0]  sm_err_code;
  logic [15:0] sm_rows, sm_cols, sm_kp;
  logic [31:0] sm_addr_s, sm_stride_s, sm_addr_p, sm_bytes_rd, sm_bytes_wr;
  logic [30:0] sm_m0;
  logic [5:0]  sm_sh;
  logic        sm_busy;
  logic [31:0] sm_readdata;

  csr_sm u_csr_sm (
    .clk, .rst_n,
    .avs_address(avs_address[5:0]), .avs_write(avs_write && sel_sm), .avs_writedata, .avs_byteenable,
    .avs_read(avs_read && sel_sm), .avs_readdata(sm_readdata),
    .start(sm_start), .soft_reset(sm_soft_reset),
    .rows(sm_rows), .cols(sm_cols), .kp(sm_kp),
    .addr_s(sm_addr_s), .stride_s(sm_stride_s), .addr_p(sm_addr_p), .m0(sm_m0), .sh(sm_sh),
    .hw_done(sm_done), .hw_err_valid(sm_err_valid), .hw_err_code(sm_err_code),
    .bytes_rd(sm_bytes_rd), .bytes_wr(sm_bytes_wr)
  );

  attn_softmax #(.LANES2(SM_LANES), .ADDR_LO(ADDR_LO), .ADDR_HI(ADDR_HI)) u_softmax (
    .clk, .rst_n, .start(sm_start), .soft_reset(sm_soft_reset),
    .rows(sm_rows), .cols(sm_cols), .kp(sm_kp),
    .s_addr(sm_addr_s), .s_stride(sm_stride_s), .p_addr(sm_addr_p), .sm_m0, .sm_sh,
    .busy(sm_busy), .done(sm_done), .err_valid(sm_err_valid), .err_code(sm_err_code),
    .bytes_rd(sm_bytes_rd), .bytes_wr(sm_bytes_wr),
    .rd_address(rd2_address), .rd_read(rd2_read), .rd_burstcount(rd2_burstcount),
    .rd_waitrequest(rd2_waitrequest), .rd_readdata(rd2_readdata), .rd_readdatavalid(rd2_readdatavalid),
    .wr_address(wr2_address), .wr_write(wr2_write), .wr_writedata(wr2_writedata),
    .wr_byteenable(wr2_byteenable), .wr_burstcount(wr2_burstcount), .wr_waitrequest(wr2_waitrequest)
  );

  assign avs_readdata = sel_sm_q ? sm_readdata : eng_readdata;
endmodule
