// Registers of the attention softmax unit (second block of the register file, byte offsets 0x100..0x1FF of
// the accelerator, i.e. this module sees word addresses 0..63). Same conventions as csr.sv: 32-bit word
// addressed slave, read latency 1, no wait states; START / RESET_SOFT / CLEAR_ERR in CTRL.
//
//   0x00 VERSION     R    'S''M' . version 0002
//   0x04 CTRL        W    [0] START  [1] RESET_SOFT  [3] CLEAR_ERR
//   0x08 STATUS      R    [0] BUSY  [1] DONE  [2] ERROR
//   0x0C ERR_CODE    R    1 START while BUSY, 2 bad parameters, 3 read timeout, 4 write timeout, 5 output FIFO overflow, 6 watchdog, 7 DMA address outside the window
//   0x10 ROWS        RW   query rows to process
//   0x14 COLS        RW   valid keys per row (1025)
//   0x18 KP          RW   k extent of P (even, >= COLS), 1026
//   0x1C ADDR_S      RW   scores (RAW32 output of the engine), 32-byte aligned
//   0x20 STRIDE_S    RW   bytes between score rows
//   0x24 ADDR_P      RW   probabilities, in the activation format of the engine
//   0x28 M0          RW   softmax_idx M0 of the block (31 bit)
//   0x2C SHIFT       RW   softmax_idx shift of the block
//   0x30 CYCLES      R    cycles of the last operation
//   0x34 BYTES_RD    R
//   0x38 BYTES_WR    R
module csr_sm #(
  parameter logic [31:0] VERSION_VALUE = 32'h534D_0002
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic [5:0]  avs_address,
  input  logic        avs_write,
  input  logic [31:0] avs_writedata,
  input  logic [3:0]  avs_byteenable,
  input  logic        avs_read,
  output logic [31:0] avs_readdata,
  // towards the unit
  output logic        start,
  output logic        soft_reset,
  output logic [15:0] rows, cols, kp,
  output logic [31:0] addr_s, stride_s, addr_p,
  output logic [30:0] m0,
  output logic [5:0]  sh,
  // from the unit
  input  logic        hw_done,
  input  logic        hw_err_valid,
  input  logic [7:0]  hw_err_code,
  input  logic [31:0] bytes_rd,
  input  logic [31:0] bytes_wr
);
  logic [31:0] rows_r, cols_r, kp_r, m0_r, sh_r;
  logic [31:0] cycles;
  logic        busy, done, error;
  logic [7:0]  err_code;

  assign rows = rows_r[15:0];
  assign cols = cols_r[15:0];
  assign kp   = kp_r[15:0];
  assign m0   = m0_r[30:0];
  assign sh   = sh_r[5:0];

  function automatic logic [31:0] merge(input logic [31:0] old, input logic [31:0] nw, input logic [3:0] be);
    logic [31:0] r;
    for (int b = 0; b < 4; b++) r[8*b +: 8] = be[b] ? nw[8*b +: 8] : old[8*b +: 8];
    return r;
  endfunction

  wire wr_ctrl = avs_write && (avs_address == 6'h01);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      start <= 1'b0; soft_reset <= 1'b0;
      rows_r <= '0; cols_r <= '0; kp_r <= '0; addr_s <= '0; stride_s <= '0; addr_p <= '0; m0_r <= '0; sh_r <= '0;
      busy <= 1'b0; done <= 1'b0; error <= 1'b0; err_code <= '0; cycles <= '0;
    end else begin
      start      <= 1'b0;
      soft_reset <= 1'b0;
      if (busy) cycles <= cycles + 1'b1;

      if (avs_write) begin
        case (avs_address)
          6'h04: rows_r   <= merge(rows_r,   avs_writedata, avs_byteenable);
          6'h05: cols_r   <= merge(cols_r,   avs_writedata, avs_byteenable);
          6'h06: kp_r     <= merge(kp_r,     avs_writedata, avs_byteenable);
          6'h07: addr_s   <= merge(addr_s,   avs_writedata, avs_byteenable);
          6'h08: stride_s <= merge(stride_s, avs_writedata, avs_byteenable);
          6'h09: addr_p   <= merge(addr_p,   avs_writedata, avs_byteenable);
          6'h0A: m0_r     <= merge(m0_r,     avs_writedata, avs_byteenable);
          6'h0B: sh_r     <= merge(sh_r,     avs_writedata, avs_byteenable);
          default: ;
        endcase
      end

      if (hw_done && busy) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
      if (hw_err_valid) begin
        busy     <= 1'b0;
        error    <= 1'b1;
        err_code <= hw_err_code;
      end

      if (wr_ctrl && avs_byteenable[0]) begin
        if (avs_writedata[3]) begin
          error    <= 1'b0;
          err_code <= '0;
        end
        if (avs_writedata[1]) begin
          soft_reset <= 1'b1;
          busy <= 1'b0; done <= 1'b0; error <= 1'b0; err_code <= '0; cycles <= '0;
        end else if (avs_writedata[0]) begin
          if (busy) begin
            error    <= 1'b1;
            err_code <= 8'd1;
          end else begin
            start  <= 1'b1;
            busy   <= 1'b1;
            done   <= 1'b0;
            cycles <= '0;
          end
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (avs_read) begin
      case (avs_address)
        6'h00: avs_readdata <= VERSION_VALUE;
        6'h01: avs_readdata <= 32'b0;
        6'h02: avs_readdata <= {29'b0, error, done, busy};
        6'h03: avs_readdata <= {24'b0, err_code};
        6'h04: avs_readdata <= rows_r;
        6'h05: avs_readdata <= cols_r;
        6'h06: avs_readdata <= kp_r;
        6'h07: avs_readdata <= addr_s;
        6'h08: avs_readdata <= stride_s;
        6'h09: avs_readdata <= addr_p;
        6'h0A: avs_readdata <= m0_r;
        6'h0B: avs_readdata <= sh_r;
        6'h0C: avs_readdata <= cycles;
        6'h0D: avs_readdata <= bytes_rd;
        6'h0E: avs_readdata <= bytes_wr;
        default: avs_readdata <= 32'hDEAD_BEEF;
      endcase
    end
  end
endmodule
