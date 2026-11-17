// Accelerator core: two register blocks behind one Avalon-MM slave, the GEMM engine and the attention
// softmax unit, each with its own pair of internal DMA masters; mem_arbiter merges the four onto one
// Avalon-MM master (mem_*) towards the HPS SDRAM port.
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
  // single master towards the HPS SDRAM port (the four DMA masters go through mem_arbiter)
  output logic [31:0]  mem_address,
  output logic         mem_read,
  output logic         mem_write,
  output logic [255:0] mem_writedata,
  output logic [31:0]  mem_byteenable,
  output logic [7:0]   mem_burstcount,
  input  logic         mem_waitrequest,
  input  logic [255:0] mem_readdata,
  input  logic         mem_readdatavalid
);
  // ---- internal DMA masters and their arbiter --------------------------------------------------------------
  logic [31:0]  rd_address, wr_address, wr_byteenable, rd2_address, wr2_address, wr2_byteenable;
  logic         rd_read, rd_waitrequest, rd_readdatavalid, wr_write, wr_waitrequest;
  logic         rd2_read, rd2_waitrequest, rd2_readdatavalid, wr2_write, wr2_waitrequest;
  logic [7:0]   rd_burstcount, wr_burstcount, rd2_burstcount, wr2_burstcount;
  logic [255:0] rd_readdata, wr_writedata, rd2_readdata, wr2_writedata, arb_readdata;

  // port 0 engine read, 1 engine write, 2 softmax read, 3 softmax write
  logic [31:0]  a_address [4];
  logic         a_read [4], a_write [4], a_wait [4], a_rdv [4];
  logic [255:0] a_wdata [4];
  logic [31:0]  a_be [4];
  logic [7:0]   a_bc [4];
  always_comb begin
    a_address[0] = rd_address;   a_read[0] = rd_read;   a_write[0] = 1'b0;      a_wdata[0] = '0;           a_be[0] = '0;              a_bc[0] = rd_burstcount;
    a_address[1] = wr_address;   a_read[1] = 1'b0;      a_write[1] = wr_write;  a_wdata[1] = wr_writedata;  a_be[1] = wr_byteenable;   a_bc[1] = wr_burstcount;
    a_address[2] = rd2_address;  a_read[2] = rd2_read;  a_write[2] = 1'b0;      a_wdata[2] = '0;           a_be[2] = '0;              a_bc[2] = rd2_burstcount;
    a_address[3] = wr2_address;  a_read[3] = 1'b0;      a_write[3] = wr2_write; a_wdata[3] = wr2_writedata; a_be[3] = wr2_byteenable;  a_bc[3] = wr2_burstcount;
    rd_waitrequest  = a_wait[0];  rd_readdatavalid  = a_rdv[0];
    wr_waitrequest  = a_wait[1];
    rd2_waitrequest = a_wait[2];  rd2_readdatavalid = a_rdv[2];
    wr2_waitrequest = a_wait[3];
    rd_readdata = arb_readdata;   rd2_readdata = arb_readdata;
  end

  mem_arbiter u_arb (
    .clk, .rst_n,
    .m_address(a_address), .m_read(a_read), .m_write(a_write), .m_writedata(a_wdata),
    .m_byteenable(a_be), .m_burstcount(a_bc), .m_waitrequest(a_wait),
    .m_readdata(arb_readdata), .m_readdatavalid(a_rdv),
    .s_address(mem_address), .s_read(mem_read), .s_write(mem_write), .s_writedata(mem_writedata),
    .s_byteenable(mem_byteenable), .s_burstcount(mem_burstcount), .s_waitrequest(mem_waitrequest),
    .s_readdata(mem_readdata), .s_readdatavalid(mem_readdatavalid)
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
  logic [31:0] out_tstride, head_tiles, head_stride, row0, rope_cfg, rope_tab_data;
  logic        rope_tab_we;
  logic [31:0] sw_m0, sw_shift, sw_tab_data;
  logic        sw_tab_we;

  csr u_csr (
    .clk, .rst_n,
    .avs_address(avs_address[5:0]), .avs_write(avs_write && !sel_sm), .avs_writedata, .avs_byteenable,
    .avs_read(avs_read && !sel_sm), .avs_readdata(eng_readdata), .irq,
    .start, .soft_reset, .op, .dim_m, .dim_n, .dim_k,
    .addr_a, .addr_w, .addr_out, .addr_bias, .addr_scale, .stride_a, .stride_out,
    .out_tstride, .head_tiles, .head_stride, .row0, .rope_cfg, .rope_tab_we, .rope_tab_data, .sw_m0, .sw_shift, .sw_tab_we, .sw_tab_data,
    .hw_done, .hw_err_valid, .hw_err_code, .bytes_rd, .bytes_wr
  );

  gemm_engine #(.A_BEATS(A_BEATS), .W_BEATS(W_BEATS), .LANES(LANES), .ADDR_LO(ADDR_LO), .ADDR_HI(ADDR_HI)) u_engine (
    .clk, .rst_n, .start, .soft_reset,
    .op, .dim_m, .dim_n, .dim_k, .addr_a, .addr_w, .addr_out, .stride_out,
    .out_tstride, .head_tiles, .head_stride, .row0, .rope_cfg, .rope_tab_we, .rope_tab_data, .sw_m0, .sw_shift, .sw_tab_we, .sw_tab_data,
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
