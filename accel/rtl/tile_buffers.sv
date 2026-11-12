// On-chip buffers of the engine (inferred as M10K, 256-bit wide, 1-cycle read):
//  - A panel RAM: activations in "tile-major" layout (one 16-byte word per k step).
//  - Two weight buffers (ping-pong): K words of 16 bytes, plus 16 sets of epilogue
//    constants (bias, M0, shift) per buffer. While one buffer feeds the array, the DMA
//    fills the other one.
// Words are 128 bit; the RAMs are 256 bit (a DMA beat = 2 words), so a read picks a
// half of the 256-bit entry one cycle later.
module tile_buffers
  import fpgate_pkg::*;
#(
  parameter int A_BEATS = 8192,     // 256-bit entries in the A RAM (256 KiB)
  parameter int W_BEATS = 1024,     // 256-bit entries per weight buffer (K up to 2048)
  parameter int COLS    = PE_COLS,
  parameter int WR_PIPE = 2         // register stages on the write side (see below)
)(
  input  logic                            clk,
  // A panel RAM
  input  logic                            a_we,
  input  logic [$clog2(A_BEATS)-1:0]      a_waddr,
  input  logic [255:0]                    a_wdata,
  input  logic [$clog2(A_BEATS):0]        a_raddr,      // 128-bit word index
  output logic [127:0]                    a_rdata,      // valid one cycle after a_raddr
  // weight buffers: load side
  input  logic                            w_load_sel,
  input  logic                            w_we,
  input  logic [$clog2(W_BEATS)-1:0]      w_waddr,
  input  logic [255:0]                    w_wdata,
  input  logic                            c_we,         // constants beat: entries 2*c_idx, 2*c_idx+1
  input  logic [2:0]                      c_idx,
  input  logic [255:0]                    c_wdata,
  // weight buffers: compute side
  input  logic                            w_comp_sel,
  input  logic [$clog2(W_BEATS):0]        w_raddr,      // 128-bit word index
  output logic [127:0]                    w_rdata,
  // epilogue side: the constants of the tile whose rows are draining (it may lag w_comp_sel)
  input  logic                            ep_sel,
  output logic signed [COLS-1:0][31:0]    bias,
  output logic        [COLS-1:0][30:0]    m0,
  output logic        [COLS-1:0][5:0]     shift
);
  // ---- write-side pipeline -------------------------------------------------------
  // The RAMs are large (hundreds of M10K spread over the chip), so address, data and
  // enables arrive through WR_PIPE register stages. The engine delays the "DMA done"
  // event by the same amount.
  logic                       a_we_p     [WR_PIPE];
  logic [$clog2(A_BEATS)-1:0] a_waddr_p  [WR_PIPE];
  logic [255:0]               a_wdata_p  [WR_PIPE];
  logic                       w_we_p     [WR_PIPE];
  logic [$clog2(W_BEATS)-1:0] w_waddr_p  [WR_PIPE];
  logic [255:0]               w_wdata_p  [WR_PIPE];
  logic                       w_lsel_p   [WR_PIPE];
  logic                       c_we_p     [WR_PIPE];
  logic [2:0]                 c_idx_p    [WR_PIPE];
  logic [255:0]               c_wdata_p  [WR_PIPE];

  always_ff @(posedge clk) begin
    a_we_p[0] <= a_we;  a_waddr_p[0] <= a_waddr;  a_wdata_p[0] <= a_wdata;
    w_we_p[0] <= w_we;  w_waddr_p[0] <= w_waddr;  w_wdata_p[0] <= w_wdata;  w_lsel_p[0] <= w_load_sel;
    c_we_p[0] <= c_we;  c_idx_p[0]   <= c_idx;    c_wdata_p[0] <= c_wdata;
    for (int s = 1; s < WR_PIPE; s++) begin
      a_we_p[s] <= a_we_p[s-1];  a_waddr_p[s] <= a_waddr_p[s-1];  a_wdata_p[s] <= a_wdata_p[s-1];
      w_we_p[s] <= w_we_p[s-1];  w_waddr_p[s] <= w_waddr_p[s-1];  w_wdata_p[s] <= w_wdata_p[s-1];  w_lsel_p[s] <= w_lsel_p[s-1];
      c_we_p[s] <= c_we_p[s-1];  c_idx_p[s]   <= c_idx_p[s-1];    c_wdata_p[s] <= c_wdata_p[s-1];
    end
  end

  wire                       a_we_q    = a_we_p[WR_PIPE-1];
  wire [$clog2(A_BEATS)-1:0] a_waddr_q = a_waddr_p[WR_PIPE-1];
  wire [255:0]               a_wdata_q = a_wdata_p[WR_PIPE-1];
  wire                       w_we_q    = w_we_p[WR_PIPE-1];
  wire [$clog2(W_BEATS)-1:0] w_waddr_q = w_waddr_p[WR_PIPE-1];
  wire [255:0]               w_wdata_q = w_wdata_p[WR_PIPE-1];
  wire                       w_lsel_q  = w_lsel_p[WR_PIPE-1];
  wire                       c_we_q    = c_we_p[WR_PIPE-1];
  wire [2:0]                 c_idx_q   = c_idx_p[WR_PIPE-1];
  wire [255:0]               c_wdata_q = c_wdata_p[WR_PIPE-1];

  // ---- A RAM ----------------------------------------------------------------
  logic [255:0] a_mem [A_BEATS];
  logic [255:0] a_q;
  logic         a_half;

  always_ff @(posedge clk) begin
    if (a_we_q) a_mem[a_waddr_q] <= a_wdata_q;
    a_q    <= a_mem[a_raddr[$clog2(A_BEATS):1]];
    a_half <= a_raddr[0];
  end
  assign a_rdata = a_half ? a_q[255:128] : a_q[127:0];

  // ---- weight buffers ---------------------------------------------------------
  logic [255:0] w_mem0 [W_BEATS];
  logic [255:0] w_mem1 [W_BEATS];
  logic [255:0] w_q0, w_q1;
  logic         w_half, w_sel_q;

  always_ff @(posedge clk) begin
    if (w_we_q && !w_lsel_q) w_mem0[w_waddr_q] <= w_wdata_q;
    if (w_we_q &&  w_lsel_q) w_mem1[w_waddr_q] <= w_wdata_q;
    w_q0    <= w_mem0[w_raddr[$clog2(W_BEATS):1]];
    w_q1    <= w_mem1[w_raddr[$clog2(W_BEATS):1]];
    w_half  <= w_raddr[0];
    w_sel_q <= w_comp_sel;
  end
  logic [255:0] w_q;
  assign w_q     = w_sel_q ? w_q1 : w_q0;
  assign w_rdata = w_half ? w_q[255:128] : w_q[127:0];

  // ---- epilogue constants, one set per buffer -----------------------------------
  logic signed [31:0] c_bias  [2][COLS];
  logic        [30:0] c_m0    [2][COLS];
  logic        [5:0]  c_shift [2][COLS];

  always_ff @(posedge clk) begin
    if (c_we_q) begin
      c_bias [w_lsel_q][{c_idx_q, 1'b0}] <= c_wdata_q[31:0];
      c_m0   [w_lsel_q][{c_idx_q, 1'b0}] <= c_wdata_q[62:32];
      c_shift[w_lsel_q][{c_idx_q, 1'b0}] <= c_wdata_q[69:64];
      c_bias [w_lsel_q][{c_idx_q, 1'b1}] <= c_wdata_q[159:128];
      c_m0   [w_lsel_q][{c_idx_q, 1'b1}] <= c_wdata_q[190:160];
      c_shift[w_lsel_q][{c_idx_q, 1'b1}] <= c_wdata_q[197:192];
    end
  end

  always_comb begin
    for (int j = 0; j < COLS; j++) begin
      bias[j]  = c_bias [ep_sel][j];
      m0[j]    = c_m0   [ep_sel][j];
      shift[j] = c_shift[ep_sel][j];
    end
  end
endmodule
