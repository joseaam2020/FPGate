// Memory arbiter: four Avalon-MM masters (256-bit data, byte addresses) onto one master towards the
// HPS SDRAM port (f2h_sdram0). Replaces the Platform Designer interconnect that sat in front of the port.
//
//   port 0: engine read     port 1: engine write     port 2: softmax read     port 3: softmax write
//
// Each port may issue reads and/or writes, but not both in the same cycle (the DMA masters here are either
// read-only or write-only). Rules:
//  - round-robin by whole burst: the pointer moves to the next port after every accepted command; a write
//    burst of N beats keeps the bus until its last beat (the other ports see waitrequest meanwhile).
//  - reads: up to DEPTH bursts outstanding. A FIFO holds {port, burstcount} of every accepted read in issue
//    order and routes each readdatavalid beat to the owner of the oldest burst. Masters never stall readdata
//    (it is broadcast to all four).
//  - each port has a one-entry input register (waitrequest low = command captured; it depends on registers only).
//  - every output towards the HPS port comes from a register. Two stages (output register + skid register)
//    so that a master's waitrequest depends only on registered state (not on the HPS waitrequest) and a
//    command still goes out every cycle when the HPS does not stall.
// A read command must carry burstcount >= 1 (0 would never return data and is not supported).
module mem_arbiter #(
  parameter int DEPTH = 16                 // outstanding read bursts
)(
  input  logic         clk,
  input  logic         rst_n,
  // masters (index = port)
  input  logic [31:0]  m_address     [4],
  input  logic         m_read        [4],
  input  logic         m_write       [4],
  input  logic [255:0] m_writedata   [4],
  input  logic [31:0]  m_byteenable  [4],
  input  logic [7:0]   m_burstcount  [4],
  output logic         m_waitrequest [4],
  output logic [255:0] m_readdata,
  output logic         m_readdatavalid [4],
  // towards the HPS port
  output logic [31:0]  s_address,
  output logic         s_read,
  output logic         s_write,
  output logic [255:0] s_writedata,
  output logic [31:0]  s_byteenable,
  output logic [7:0]   s_burstcount,
  input  logic         s_waitrequest,
  input  logic [255:0] s_readdata,
  input  logic         s_readdatavalid
);
  localparam int PW = $clog2(DEPTH);

  // ---- command stages -----------------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0]  address;
    logic         rd;
    logic         wr;
    logic [255:0] data;
    logic [31:0]  be;
    logic [7:0]   bc;
  } cmd_t;

  // per-port input register: a master's command is captured here, so the master's registered outputs
  // (e.g. dma_wr's pending) do not feed the grant logic and the data mux in the same cycle
  cmd_t q_cmd [4];
  logic [3:0] q_v;

  cmd_t p_cmd, s_cmd;                       // output register, skid register
  logic p_valid, s_valid;

  assign s_address    = p_cmd.address;
  assign s_read       = p_valid & p_cmd.rd;
  assign s_write      = p_valid & p_cmd.wr;
  assign s_writedata  = p_cmd.data;
  assign s_byteenable = p_cmd.be;
  assign s_burstcount = p_cmd.bc;

  // ---- read ID FIFO ---------------------------------------------------------------------------------------
  logic [9:0]   id_mem [DEPTH];             // {port[1:0], burstcount[7:0]}
  logic [PW-1:0] id_wp, id_rp;
  logic [PW:0]   id_cnt;
  logic [7:0]    beat;                      // beats already returned of the oldest burst
  wire  rd_room = (id_cnt != (PW+1)'(DEPTH));
  wire [9:0] id_head = id_mem[id_rp];

  // ---- arbitration ----------------------------------------------------------------------------------------
  logic [1:0] rr;                           // port granted last (priority starts after it)
  logic       lock;                         // inside a multi-beat write burst
  logic [1:0] lock_port;
  logic [7:0] lock_left;                    // beats still to come after the current one

  logic [3:0] req;
  always_comb
    for (int i = 0; i < 4; i++)
      req[i] = q_v[i] && (q_cmd[i].wr || rd_room);

  logic       gnt_valid;
  logic [1:0] gnt;
  always_comb begin
    gnt_valid = 1'b0;
    gnt       = rr;
    if (lock) begin
      gnt_valid = req[lock_port];
      gnt       = lock_port;
    end else begin
      for (int k = 3; k >= 0; k--) begin    // last assignment wins: k = 0 (the port after rr) has priority
        if (req[2'(rr + 2'(k) + 2'd1)]) begin
          gnt_valid = 1'b1;
          gnt       = 2'(rr + 2'(k) + 2'd1);
        end
      end
    end
  end

  wire take = gnt_valid && !s_valid;        // a command is accepted this cycle (depends on registers only)

  always_comb
    for (int i = 0; i < 4; i++)
      m_waitrequest[i] = (q_v[i] && !(take && gnt == 2'(i))) || !rst_n;

  cmd_t in_cmd;
  assign in_cmd = q_cmd[gnt];

  always_ff @(posedge clk) begin
    if (!rst_n) q_v <= '0;
    else begin
      for (int i = 0; i < 4; i++) begin
        if (!m_waitrequest[i] && (m_read[i] || m_write[i])) begin
          q_cmd[i].address <= m_address[i];
          q_cmd[i].wr      <= m_write[i];
          q_cmd[i].rd      <= !m_write[i] && m_read[i];
          q_cmd[i].data    <= m_writedata[i];
          q_cmd[i].be      <= m_byteenable[i];
          q_cmd[i].bc      <= m_burstcount[i];
          q_v[i]           <= 1'b1;
        end else if (take && gnt == 2'(i)) begin
          q_v[i] <= 1'b0;
        end
      end
    end
  end

  wire p_taken = p_valid && !s_waitrequest;
  wire acc_rd  = take && in_cmd.rd;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      p_valid <= 1'b0;
      s_valid <= 1'b0;
      lock    <= 1'b0;
      rr      <= 2'd3;
      id_wp   <= '0;
      id_rp   <= '0;
      id_cnt  <= '0;
      beat    <= '0;
    end else begin
      // command path
      if (p_taken || !p_valid) begin
        if (s_valid) begin
          p_cmd   <= s_cmd;
          p_valid <= 1'b1;
          s_valid <= 1'b0;
        end else begin
          p_cmd   <= in_cmd;
          p_valid <= take;
        end
      end else if (take) begin
        s_cmd   <= in_cmd;
        s_valid <= 1'b1;
      end

      // grant bookkeeping
      if (take) begin
        if (in_cmd.wr) begin
          if (lock) begin
            lock_left <= lock_left - 1'b1;
            if (lock_left == 8'd1) begin
              lock <= 1'b0;
              rr   <= gnt;
            end
          end else if (in_cmd.bc > 8'd1) begin
            lock      <= 1'b1;
            lock_port <= gnt;
            lock_left <= in_cmd.bc - 8'd1;
          end else begin
            rr <= gnt;
          end
        end else begin
          rr <= gnt;
        end
      end

      // read ID FIFO
      if (acc_rd) begin
        id_mem[id_wp] <= {gnt, in_cmd.bc};
        id_wp         <= id_wp + 1'b1;
      end
      if (s_readdatavalid) begin
        if (beat + 1'b1 == id_head[7:0]) begin
          beat  <= '0;
          id_rp <= id_rp + 1'b1;
        end else begin
          beat <= beat + 1'b1;
        end
      end
      id_cnt <= id_cnt + (PW+1)'(acc_rd) - (PW+1)'(s_readdatavalid && (beat + 1'b1 == id_head[7:0]));
    end
  end

  assign m_readdata = s_readdata;
  always_comb
    for (int i = 0; i < 4; i++)
      m_readdatavalid[i] = s_readdatavalid && (id_head[9:8] == 2'(i));
endmodule
