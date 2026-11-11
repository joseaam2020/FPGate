// Read DMA: Avalon-MM master (256-bit data, byte addresses) that fetches a run of
// beats and streams them out in order (beat_valid / beat_data / beat_idx). The
// consumer must accept every beat (it writes them straight into an on-chip RAM).
//
// Bursts are at most BURST beats and never cross a 4 KiB boundary. If a burst gets
// no data for TIMEOUT cycles the command is aborted with `err`.
module dma_rd #(
  parameter int BURST   = 16,
  parameter int TIMEOUT = 4096
)(
  input  logic         clk,
  input  logic         rst_n,
  // command
  input  logic         cmd_valid,
  output logic         cmd_ready,
  input  logic [31:0]  cmd_addr,        // byte address, 32-byte aligned
  input  logic [15:0]  cmd_beats,
  output logic         busy,
  output logic         done,            // 1-cycle pulse
  output logic         err,             // 1-cycle pulse (timeout)
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
  typedef enum logic [1:0] {S_IDLE, S_PREP, S_REQ, S_DATA} state_t;
  state_t state;

  logic [31:0] addr;
  logic [15:0] left;
  logic [7:0]  bl, got;
  logic [31:0] tmo;
  logic [15:0] idx;

  // burst length: min(left, BURST, beats until the next 4 KiB boundary)
  logic [7:0] to_bound, bl_next;
  always_comb begin
    to_bound = 8'(8'd128 - {1'b0, addr[11:5]});
    bl_next  = (left < 16'(BURST)) ? left[7:0] : 8'(BURST);
    if (to_bound < bl_next) bl_next = to_bound;
  end

  assign cmd_ready    = (state == S_IDLE);
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
    if (!rst_n) begin
      state <= S_IDLE;
      idx   <= '0;
    end else begin
      case (state)
        S_IDLE: if (cmd_valid) begin
          addr <= cmd_addr;
          left <= cmd_beats;
          idx  <= '0;
          if (cmd_beats == 0) done <= 1'b1;
          else                state <= S_PREP;
        end
        S_PREP: begin
          bl    <= bl_next;
          state <= S_REQ;
        end
        S_REQ: if (!m_waitrequest) begin
          got   <= '0;
          tmo   <= '0;
          state <= S_DATA;
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
              if (left == 16'(bl)) begin
                done  <= 1'b1;
                state <= S_IDLE;
              end else begin
                state <= S_PREP;
              end
            end
          end else if (tmo >= 32'(TIMEOUT)) begin
            err   <= 1'b1;
            state <= S_IDLE;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
