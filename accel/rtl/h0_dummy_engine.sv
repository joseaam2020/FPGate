// Placeholder engine for H0 (register bring-up on the board). It does no compute:
// after START it stays busy for DIM_M cycles (1000 if DIM_M = 0) and then reports
// done, or reports an error (code 0x10) if OP[31] is set. This lets software check
// the START/BUSY/DONE/ERROR flow and the cycle counter before the real engine exists.
// Replaced by the tile sequencer in H2.
module h0_dummy_engine (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic        soft_reset,
  input  logic [31:0] op,
  input  logic [31:0] dim_m,
  output logic        hw_done,
  output logic        hw_err_valid,
  output logic [7:0]  hw_err_code
);
  logic [31:0] left;
  logic        running, fail;

  always_ff @(posedge clk) begin
    hw_done      <= 1'b0;
    hw_err_valid <= 1'b0;
    hw_err_code  <= 8'h10;
    if (!rst_n || soft_reset) begin
      running <= 1'b0;
      left    <= '0;
      fail    <= 1'b0;
    end else if (start) begin
      running <= 1'b1;
      left    <= (dim_m == 0) ? 32'd1000 : dim_m;
      fail    <= op[31];
    end else if (running) begin
      if (left <= 1) begin
        running <= 1'b0;
        if (fail) hw_err_valid <= 1'b1;
        else      hw_done      <= 1'b1;
      end else begin
        left <= left - 1'b1;
      end
    end
  end
endmodule
