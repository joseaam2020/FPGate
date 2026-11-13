// Write DMA: accepts one 256-bit beat at a time (address, data, byte enables) and
// writes it with a single-beat Avalon-MM write. valid/ready on the input side; a new
// beat can be taken in the same cycle the previous one completes. If the slave holds
// waitrequest for TIMEOUT cycles `err` pulses once.
//
// A beat that was presented to the bus is never withdrawn: dropping `write` while the slave
// still holds waitrequest breaks the Avalon rule and can leave the f2h interconnect waiting
// for the rest of the transaction (the port then moves 0 bytes until reset). On `flush` (soft
// reset / abort of the operation) and after a timeout the beat is therefore held, unchanged,
// until the slave takes it. It is then "stale": no wr_beat/err is reported for it. The
// burst length is 1, so there are no missing beats of a started burst to pad with
// byteenable = 0.
//
// Address window guard: a beat outside [ADDR_LO, ADDR_HI) is consumed but never put on the bus;
// `err_win` pulses once. The check only feeds the registers, not the Avalon port.
module dma_wr #(
  parameter int TIMEOUT = 4096,
  parameter logic [31:0] ADDR_LO = 32'h3000_0000,
  parameter logic [31:0] ADDR_HI = 32'h4000_0000
)(
  input  logic         clk,
  input  logic         rst_n,          // hard reset only (drops a pending beat)
  input  logic         flush,          // abandon the operation: refuse new beats, finish the pending one
  input  logic         in_valid,
  output logic         in_ready,
  input  logic [31:0]  in_addr,        // byte address, 32-byte aligned
  input  logic [255:0] in_data,
  input  logic [31:0]  in_be,
  output logic         busy,
  output logic         wr_beat,        // 1-cycle pulse per beat written
  output logic         err,            // 1-cycle pulse
  output logic         err_win,        // 1-cycle pulse (beat outside [ADDR_LO, ADDR_HI))
  // Avalon-MM master (write only)
  output logic [31:0]  m_address,
  output logic         m_write,
  output logic [255:0] m_writedata,
  output logic [31:0]  m_byteenable,
  output logic [7:0]   m_burstcount,
  input  logic         m_waitrequest
);
  logic        pending;
  logic        stale;
  logic [31:0] tmo;

  wire win_ok = (in_addr >= ADDR_LO) && ({1'b0, in_addr} + 33'd32 <= {1'b0, ADDR_HI});

  assign m_write      = pending;
  assign m_burstcount = 8'd1;
  assign busy         = pending;
  assign in_ready     = (!pending || !m_waitrequest) && !flush;

  always_ff @(posedge clk) begin
    wr_beat <= 1'b0;
    err     <= 1'b0;
    err_win <= 1'b0;
    if (!rst_n) begin
      pending <= 1'b0;
      stale   <= 1'b0;
      tmo     <= '0;
    end else begin
      if (pending && flush) stale <= 1'b1;
      if (pending && m_waitrequest) begin
        tmo <= tmo + 1'b1;
        if (tmo >= 32'(TIMEOUT) && !stale && !flush) begin
          err   <= 1'b1;
          stale <= 1'b1;          // keep holding the beat; report nothing more for it
        end
      end else begin
        tmo <= '0;
      end
      if (pending && !m_waitrequest) begin
        pending <= 1'b0;
        stale   <= 1'b0;
        wr_beat <= !stale && !flush;
      end
      if (in_valid && in_ready) begin
        m_address    <= in_addr;
        m_writedata  <= in_data;
        m_byteenable <= in_be;
        pending      <= win_ok;
        err_win      <= !win_ok;
      end
    end
  end
endmodule
