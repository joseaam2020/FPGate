// Write DMA: accepts one 256-bit beat at a time (address, data, byte enables) and
// writes it with a single-beat Avalon-MM write. valid/ready on the input side; a new
// beat can be taken in the same cycle the previous one completes. If the slave holds
// waitrequest for TIMEOUT cycles the beat is dropped and `err` pulses.
module dma_wr #(
  parameter int TIMEOUT = 4096
)(
  input  logic         clk,
  input  logic         rst_n,
  input  logic         in_valid,
  output logic         in_ready,
  input  logic [31:0]  in_addr,        // byte address, 32-byte aligned
  input  logic [255:0] in_data,
  input  logic [31:0]  in_be,
  output logic         busy,
  output logic         wr_beat,        // 1-cycle pulse per beat written
  output logic         err,            // 1-cycle pulse
  // Avalon-MM master (write only)
  output logic [31:0]  m_address,
  output logic         m_write,
  output logic [255:0] m_writedata,
  output logic [31:0]  m_byteenable,
  output logic [7:0]   m_burstcount,
  input  logic         m_waitrequest
);
  logic        pending;
  logic [31:0] tmo;

  assign m_write      = pending;
  assign m_burstcount = 8'd1;
  assign busy         = pending;
  assign in_ready     = !pending || !m_waitrequest;

  always_ff @(posedge clk) begin
    wr_beat <= 1'b0;
    err     <= 1'b0;
    if (!rst_n) begin
      pending <= 1'b0;
      tmo     <= '0;
    end else begin
      if (pending && !m_waitrequest) begin
        pending <= 1'b0;
        wr_beat <= 1'b1;
      end
      if (pending && m_waitrequest) begin
        tmo <= tmo + 1'b1;
        if (tmo >= 32'(TIMEOUT)) begin
          pending <= 1'b0;
          err     <= 1'b1;
        end
      end else begin
        tmo <= '0;
      end
      if (in_valid && in_ready) begin
        m_address    <= in_addr;
        m_writedata  <= in_data;
        m_byteenable <= in_be;
        pending      <= 1'b1;
      end
    end
  end
endmodule
