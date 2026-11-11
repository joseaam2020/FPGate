// Accelerator top level (H0): CSR + placeholder engine behind an Avalon-MM slave.
// Port names follow the Avalon naming (avs_*) so Platform Designer can take this
// module as a custom component. The real engine (tile sequencer, DMA, array,
// epilogue) replaces h0_dummy_engine in later milestones.
module fpgate_accel_top (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [5:0]  avs_address,
  input  logic        avs_write,
  input  logic [31:0] avs_writedata,
  input  logic [3:0]  avs_byteenable,
  input  logic        avs_read,
  output logic [31:0] avs_readdata,
  output logic        irq
);
  logic        start, soft_reset, hw_done, hw_err_valid;
  logic [7:0]  hw_err_code;
  logic [31:0] op, dim_m, dim_n, dim_k, addr_a, addr_w, addr_out, addr_bias, addr_scale, stride_a, stride_out;

  csr u_csr (
    .clk, .rst_n,
    .avs_address, .avs_write, .avs_writedata, .avs_byteenable, .avs_read, .avs_readdata, .irq,
    .start, .soft_reset, .op, .dim_m, .dim_n, .dim_k,
    .addr_a, .addr_w, .addr_out, .addr_bias, .addr_scale, .stride_a, .stride_out,
    .hw_done, .hw_err_valid, .hw_err_code,
    .bytes_rd(32'd0), .bytes_wr(32'd0)
  );

  h0_dummy_engine u_engine (
    .clk, .rst_n, .start, .soft_reset, .op, .dim_m,
    .hw_done, .hw_err_valid, .hw_err_code
  );
endmodule
