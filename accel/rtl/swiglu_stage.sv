// SwiGLU stage between the epilogue and the output rows (OP.SWIGLU), for fc1 with interleaved weights.
//
// The operation walks tiles gate(0), up(0), gate(1), up(1), ... (16 channels each). Per panel (up to SUBS subtiles
// of 16 rows) all gate(j) subtiles come out first, then all up(j) subtiles. The epilogue requantizes both to INT8
// (gi, ui), exactly as the two separate fc1g / fc1x operations would. This stage
//   * stores the 16 x 16 INT8 values of every gate subtile in a RAM (16 subtiles x 16 rows x 16 bytes = 4 KiB);
//   * for every up row reads the matching gate row and runs the 16 values through swiglu_lane (LANES values per
//     cycle, so a row takes 16 / LANES cycles), which gives h = sat16((silu[gi] * ui * M0 + 2^(shift-1)) >> shift),
//     the same arithmetic as the software;
//   * emits one row per input row, in order: up rows carry h, gate rows are emitted with out_skip = 1 and no data
//     (they keep the output side's subtile bookkeeping unchanged, which counts rows per subtile).
// Rows enter an 8-deep FIFO (credit `room`, registered by the caller, leaves 5 rows for the ones already in the
// epilogue lanes) and leave one at a time, LANES = 1 being the cheapest (one swiglu_lane, one SiLU table):
// a row takes 16 cycles, a subtile 256, while the array needs K (>= 256) cycles per subtile.
//
// The subtile context (gate or up, subtile index in the panel) comes with sub_start from the sequencer and is
// attached to the rows of that subtile as they come in.
// Multipliers in logic: the DSP blocks are all taken by the GEMM array and the epilogue.
(* multstyle = "logic" *) module swiglu_stage
  import fpgate_pkg::*;
#(
  parameter int COLS   = 16,
  parameter int LANES  = 1,          // values per cycle (divides COLS); every lane has its own SiLU table
  parameter int FIFO_D = 8,
  parameter int SUBS   = 16          // subtiles per panel the gate buffer holds
)(
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          en,
  // block constants
  input  logic        [30:0]            m0,
  input  logic        [5:0]             shift,
  input  logic                          tbl_we,
  input  logic        [7:0]             tbl_addr,
  input  logic signed [15:0]            tbl_data,
  // subtile context (pulse with the sequencer's sub_start)
  input  logic                          ctx_push,
  input  logic                          ctx_up,
  input  logic        [3:0]             ctx_sidx,
  // INT8 rows from the epilogue (value in bits [7:0] of every element)
  input  logic                          in_valid,
  input  logic signed [COLS-1:0][15:0]  in_data,
  input  logic                          in_last,
  output logic                          room,        // combinational: the FIFO can take more rows (caller registers it)
  input  logic                          out_ok,      // the consumer has room for the rows in flight
  // rows out
  output logic                          out_valid,
  output logic signed [COLS-1:0][15:0]  out_data,
  output logic                          out_last,
  output logic                          out_skip
);
  localparam int NB  = COLS / LANES;                 // beats per row
  localparam int BW  = (NB > 1) ? $clog2(NB) : 1;
  localparam int LAT = 3 + RQ_LATENCY;               // swiglu_lane latency
  localparam int FW  = $clog2(FIFO_D);
  localparam int AW  = $clog2(SUBS) + 4;

  // ---- subtile contexts --------------------------------------------------------------------------
  logic        cq_up   [4];
  logic [3:0]  cq_sidx [4];
  logic [2:0]  cq_w, cq_r;
  logic [3:0]  rowcnt;
  wire         cq_empty = (cq_w == cq_r);

  // ---- input FIFO ----------------------------------------------------------------------------------
  logic [127:0]  f_data [FIFO_D];
  logic          f_up   [FIFO_D];
  logic [AW-1:0] f_addr [FIFO_D];
  logic          f_last [FIFO_D];
  logic [FW:0]   fw, fr;
  wire  [FW:0]   used  = fw - fr;
  wire           f_empty = (fw == fr);
  assign room = (32'(used) + 32'd5) <= FIFO_D;

  logic [127:0] in_bytes;
  always_comb for (int j = 0; j < COLS; j++) in_bytes[8*j +: 8] = in_data[j][7:0];

  wire take = en && in_valid && !cq_empty;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cq_w <= '0; cq_r <= '0; rowcnt <= '0;
      fw <= '0;
    end else begin
      if (en && ctx_push) begin
        cq_up[cq_w[1:0]]   <= ctx_up;
        cq_sidx[cq_w[1:0]] <= ctx_sidx;
        cq_w <= cq_w + 1'b1;
      end
      if (take) begin
        f_data[fw[FW-1:0]] <= in_bytes;
        f_up[fw[FW-1:0]]   <= cq_up[cq_r[1:0]];
        f_addr[fw[FW-1:0]] <= AW'({cq_sidx[cq_r[1:0]], rowcnt});
        f_last[fw[FW-1:0]] <= in_last;
        fw     <= fw + 1'b1;
        rowcnt <= rowcnt + 1'b1;
        if (in_last) begin
          rowcnt <= '0;
          cq_r   <= cq_r + 1'b1;
        end
      end
    end
  end

  // ---- gate buffer ----------------------------------------------------------------------------------
  logic [127:0] gram [2**AW];
  logic [127:0] ram_q;

  // ---- row processor --------------------------------------------------------------------------------
  logic            active;
  logic [BW-1:0]   beat;
  logic [127:0]    cur_ui;
  logic            cur_skip, cur_last;
  wire             last_beat = (beat == BW'(NB - 1));
  wire             pop = !f_empty && out_ok && (!active || last_beat);

  wire [127:0]   h_data = f_data[fr[FW-1:0]];
  wire           h_up   = f_up[fr[FW-1:0]];
  wire [AW-1:0]  h_addr = f_addr[fr[FW-1:0]];
  wire           h_last = f_last[fr[FW-1:0]];

  always_ff @(posedge clk) begin
    if (pop && !h_up) gram[h_addr] <= h_data;
    if (pop)          ram_q <= gram[h_addr];
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      fr     <= '0;
      active <= 1'b0;
      beat   <= '0;
    end else begin
      if (pop) begin
        fr       <= fr + 1'b1;
        active   <= 1'b1;
        beat     <= '0;
        cur_ui   <= h_data;
        cur_skip <= !h_up;
        cur_last <= h_last;
      end else if (active) begin
        beat <= beat + 1'b1;
        if (last_beat) active <= 1'b0;
      end
    end
  end

  // ---- lanes ------------------------------------------------------------------------------------------
  logic                lane_valid;
  logic signed [7:0]   l_gi [LANES];
  logic signed [7:0]   l_ui [LANES];
  logic                lane_out_valid [LANES];
  logic signed [15:0]  lane_h         [LANES];

  always_comb begin
    lane_valid = active;
    for (int l = 0; l < LANES; l++) begin
      l_ui[l] = cur_ui[8 * (beat * LANES + l) +: 8];
      l_gi[l] = cur_skip ? 8'sd0 : ram_q[8 * (beat * LANES + l) +: 8];
    end
  end

  genvar gl;
  generate
  for (gl = 0; gl < LANES; gl++) begin : g_lane
    swiglu_lane u_lane (
      .clk, .rst_n,
      .tbl_we, .tbl_addr, .tbl_data,
      .in_valid(lane_valid), .gi(l_gi[gl]), .ui(l_ui[gl]), .m0, .shift,
      .out_valid(lane_out_valid[gl]), .h(lane_h[gl])
    );
  end
  endgenerate

  // sideband follows the lane latency
  logic [BW-1:0] sb_beat [LAT];
  logic          sb_lastbeat [LAT];
  logic          sb_rowlast [LAT];
  logic          sb_skip [LAT];
  always_ff @(posedge clk) begin
    sb_beat[0]     <= beat;
    sb_lastbeat[0] <= last_beat;
    sb_rowlast[0]  <= cur_last;
    sb_skip[0]     <= cur_skip;
    for (int d = 1; d < LAT; d++) begin
      sb_beat[d]     <= sb_beat[d-1];
      sb_lastbeat[d] <= sb_lastbeat[d-1];
      sb_rowlast[d]  <= sb_rowlast[d-1];
      sb_skip[d]     <= sb_skip[d-1];
    end
  end

  // ---- collect ------------------------------------------------------------------------------------------
  logic signed [COLS-1:0][15:0] acc_row;

  always_ff @(posedge clk) begin
    out_valid <= 1'b0;
    if (lane_out_valid[0]) begin
      logic signed [COLS-1:0][15:0] nxt;
      nxt = acc_row;
      for (int l = 0; l < LANES; l++) nxt[sb_beat[LAT-1] * LANES + l] = lane_h[l];
      acc_row <= nxt;
      if (sb_lastbeat[LAT-1]) begin
        out_data  <= nxt;
        out_valid <= 1'b1;
        out_last  <= sb_rowlast[LAT-1];
        out_skip  <= sb_skip[LAT-1];
      end
    end
  end
endmodule
