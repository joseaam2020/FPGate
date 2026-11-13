// Read DMA: Avalon-MM master (256-bit data, byte addresses) that fetches a run of
// beats and streams them out in order (beat_valid / beat_data / beat_idx). The
// consumer must accept every beat (it writes them straight into an on-chip RAM).
//
// Bursts are at most BURST beats and never cross a 4 KiB boundary. If a burst gets
// no data for TIMEOUT cycles the command is aborted with `err`.
//
// Address window guard: a burst that starts below ADDR_LO or ends above ADDR_HI (exclusive) is
// never issued; `err_win` pulses once and the command is abandoned. The check is made while the
// burst is prepared (S_PREP) and only feeds the state register, not the Avalon port.
//
// On `flush` (soft reset / abort) or after a timeout the unit does not drop what is on the
// bus: a request that is waiting for the slave is held until accepted, and the beats the slave
// still owes for an accepted burst are consumed (S_DRAIN, bounded by TIMEOUT) before the unit
// is ready again, so stale data never leaks into the next command.
module dma_rd #(
  parameter int BURST   = 16,
  parameter int TIMEOUT = 4096,
  parameter logic [31:0] ADDR_LO = 32'h3000_0000,
  parameter logic [31:0] ADDR_HI = 32'h4000_0000
)(
  input  logic         clk,
  input  logic         rst_n,          // hard reset only
  input  logic         flush,          // abandon the command, finish the bus transaction
  // command
  input  logic         cmd_valid,
  output logic         cmd_ready,
  input  logic [31:0]  cmd_addr,        // byte address, 32-byte aligned
  input  logic [15:0]  cmd_beats,
  output logic         busy,
  output logic         done,            // 1-cycle pulse
  output logic         err,             // 1-cycle pulse (timeout)
  output logic         err_win,         // 1-cycle pulse (burst outside [ADDR_LO, ADDR_HI))
  // stream
  output logic         beat_valid,
  output logic [255:0] beat_data,
  output logic [15:0]  beat_idx,        // index of the beat within the command
  // Avalon-MM master (read only)
  output logic [31:0]  m_address,
  output logic         m_read,
  output logic [7:0]   m_burstcount,
  input  logic         m_waitrequest,
  input  logic [255:0] m_readdata,
  input  logic         m_readdatavalid
);
  typedef enum logic [2:0] {S_IDLE, S_PREP, S_REQ, S_DATA, S_DRAIN} state_t;
  state_t state;

  logic [31:0] addr;
  logic [15:0] left;
  logic [7:0]  bl, got;
  logic [31:0] tmo;
  logic [15:0] idx;
  logic        aborted;               // flush seen while the request was waiting

  // burst length: min(left, BURST, beats until the next 4 KiB boundary)
  logic [7:0] to_bound, bl_next;
  always_comb begin
    to_bound = 8'(8'd128 - {1'b0, addr[11:5]});
    bl_next  = (left < 16'(BURST)) ? left[7:0] : 8'(BURST);
    if (to_bound < bl_next) bl_next = to_bound;
  end

  // window check of the burst about to be requested (33-bit end so it cannot wrap)
  logic win_bad;
  always_comb begin
    win_bad = (addr < ADDR_LO) || ({1'b0, addr} + {17'b0, bl_next, 5'b0} > {1'b0, ADDR_HI});
  end

  assign cmd_ready    = (state == S_IDLE) && !flush;
  assign busy         = (state != S_IDLE);
  assign m_read       = (state == S_REQ);
  assign m_address    = addr;
  assign m_burstcount = bl;               // registered in S_PREP: no logic in front of the HPS port
  assign beat_valid   = (state == S_DATA) && m_readdatavalid;
  assign beat_data    = m_readdata;
  assign beat_idx     = idx;

  always_ff @(posedge clk) begin
    done <= 1'b0;
    err  <= 1'b0;
    err_win <= 1'b0;
    if (!rst_n) begin
      state   <= S_IDLE;
      idx     <= '0;
      aborted <= 1'b0;
    end else begin
      if (flush && state == S_REQ) aborted <= 1'b1;
      case (state)
        S_IDLE: if (cmd_valid && !flush) begin
          addr    <= cmd_addr;
          left    <= cmd_beats;
          idx     <= '0;
          aborted <= 1'b0;
          if (cmd_beats == 0) done <= 1'b1;
          else                state <= S_PREP;
        end
        S_PREP: begin
          bl    <= bl_next;
          if (flush) begin
            state <= S_IDLE;
          end else if (win_bad) begin
            err_win <= 1'b1;               // nothing was put on the bus: just abandon the command
            state   <= S_IDLE;
          end else begin
            state <= S_REQ;
          end
        end
        S_REQ: if (!m_waitrequest) begin
          got   <= '0;
          tmo   <= '0;
          state <= (aborted || flush) ? S_DRAIN : S_DATA;
        end
        S_DATA: begin
          tmo <= tmo + 1'b1;
          if (m_readdatavalid) begin
            tmo <= '0;
            got <= got + 1'b1;
            idx <= idx + 1'b1;
            if (got + 1'b1 == bl) begin
              addr <= addr + {bl, 5'b0};
              left <= left - 16'(bl);
              if (flush) begin
                state <= S_IDLE;
              end else if (left == 16'(bl)) begin
                done  <= 1'b1;
                state <= S_IDLE;
              end else begin
                state <= S_PREP;
              end
            end else if (flush) begin
              state <= S_DRAIN;
            end
          end else if (flush) begin
            tmo   <= '0;
            state <= S_DRAIN;
          end else if (tmo >= 32'(TIMEOUT)) begin
            err   <= 1'b1;
            tmo   <= '0;
            state <= S_DRAIN;
          end
        end
        S_DRAIN: begin
          tmo <= tmo + 1'b1;
          if (m_readdatavalid) begin
            tmo <= '0;
            got <= got + 1'b1;
            if (got + 1'b1 == bl) state <= S_IDLE;
          end else if (tmo >= 32'(TIMEOUT)) begin
            state <= S_IDLE;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
