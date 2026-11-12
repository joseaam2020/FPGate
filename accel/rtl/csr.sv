// Control/status registers: the only contact point with the HPS.
// Avalon-MM slave, 32 bit, word addressed, fixed read latency 1 (no waitrequest).
// Map follows guia_DES-2 section 6.7 (byte offsets). Deviation: STRIDE_OUT lives at
// 0x4C because the guide places two registers at 0x34; to be agreed with fpgate-accel.
//
//   0x00 VERSION     R    design id / version
//   0x04 CTRL        W    [0] START  [1] RESET_SOFT  [2] IRQ_EN (level)  [3] CLEAR_ERR
//   0x08 STATUS      R    [0] BUSY  [1] DONE  [2] ERROR
//   0x0C ERR_CODE    R    last error code (0 = none)
//   0x10 OP          RW   operation type and flags
//   0x14 DIM_M       RW
//   0x18 DIM_N       RW
//   0x1C DIM_K       RW
//   0x20 ADDR_A      RW   physical addresses in the reserved DDR3 region
//   0x24 ADDR_W      RW
//   0x28 ADDR_OUT    RW
//   0x2C ADDR_BIAS   RW
//   0x30 ADDR_SCALE  RW
//   0x34 STRIDE_A    RW
//   0x38 CYCLES_LO   R    cycles of the last operation (START to DONE/ERROR)
//   0x3C CYCLES_HI   R
//   0x40 BYTES_RD    R    from the engine
//   0x44 BYTES_WR    R
//   0x48 HEARTBEAT   R    free-running counter; if it stops, the accelerator is hung
//   0x4C STRIDE_OUT  RW
//   0x50 SCRATCH     RW   read/write test register
//
// Errors: 1 = START while BUSY, otherwise the code given by the engine (hw_err_code).
module csr #(
  parameter logic [31:0] VERSION_VALUE = 32'h4650_0002   // 'F''P' . version 0002
)(
  input  logic        clk,
  input  logic        rst_n,
  // Avalon-MM slave
  input  logic [5:0]  avs_address,       // word address
  input  logic        avs_write,
  input  logic [31:0] avs_writedata,
  input  logic [3:0]  avs_byteenable,
  input  logic        avs_read,
  output logic [31:0] avs_readdata,
  output logic        irq,
  // towards the engine
  output logic        start,             // 1-cycle pulse
  output logic        soft_reset,        // 1-cycle pulse
  output logic [31:0] op, dim_m, dim_n, dim_k,
  output logic [31:0] addr_a, addr_w, addr_out, addr_bias, addr_scale,
  output logic [31:0] stride_a, stride_out,
  // from the engine
  input  logic        hw_done,           // 1-cycle pulse
  input  logic        hw_err_valid,      // 1-cycle pulse
  input  logic [7:0]  hw_err_code,
  input  logic [31:0] bytes_rd,
  input  logic [31:0] bytes_wr
);
  localparam logic [7:0] ERR_START_BUSY = 8'd1;

  logic [31:0] scratch, heartbeat;
  logic [63:0] cycles;
  logic        busy, done, error, irq_en;
  logic [7:0]  err_code;

  // byte-enable merge for the read/write registers
  function automatic logic [31:0] merge(input logic [31:0] old, input logic [31:0] nw, input logic [3:0] be);
    logic [31:0] r;
    for (int b = 0; b < 4; b++) r[8*b +: 8] = be[b] ? nw[8*b +: 8] : old[8*b +: 8];
    return r;
  endfunction

  wire wr_ctrl = avs_write && (avs_address == 6'h01);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      start <= 1'b0; soft_reset <= 1'b0;
      op <= '0; dim_m <= '0; dim_n <= '0; dim_k <= '0;
      addr_a <= '0; addr_w <= '0; addr_out <= '0; addr_bias <= '0; addr_scale <= '0;
      stride_a <= '0; stride_out <= '0; scratch <= '0;
      busy <= 1'b0; done <= 1'b0; error <= 1'b0; irq_en <= 1'b0; err_code <= '0;
      cycles <= '0;
      heartbeat <= '0;
    end else begin
      start      <= 1'b0;
      soft_reset <= 1'b0;
      heartbeat  <= heartbeat + 1'b1;
      if (busy) cycles <= cycles + 1'b1;

      // register writes
      if (avs_write) begin
        case (avs_address)
          6'h04: op         <= merge(op,         avs_writedata, avs_byteenable);
          6'h05: dim_m      <= merge(dim_m,      avs_writedata, avs_byteenable);
          6'h06: dim_n      <= merge(dim_n,      avs_writedata, avs_byteenable);
          6'h07: dim_k      <= merge(dim_k,      avs_writedata, avs_byteenable);
          6'h08: addr_a     <= merge(addr_a,     avs_writedata, avs_byteenable);
          6'h09: addr_w     <= merge(addr_w,     avs_writedata, avs_byteenable);
          6'h0A: addr_out   <= merge(addr_out,   avs_writedata, avs_byteenable);
          6'h0B: addr_bias  <= merge(addr_bias,  avs_writedata, avs_byteenable);
          6'h0C: addr_scale <= merge(addr_scale, avs_writedata, avs_byteenable);
          6'h0D: stride_a   <= merge(stride_a,   avs_writedata, avs_byteenable);
          6'h13: stride_out <= merge(stride_out, avs_writedata, avs_byteenable);
          6'h14: scratch    <= merge(scratch,    avs_writedata, avs_byteenable);
          default: ;
        endcase
      end

      // engine events
      if (hw_done && busy) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
      if (hw_err_valid) begin
        busy     <= 1'b0;
        error    <= 1'b1;
        err_code <= hw_err_code;
      end

      // control register (bytes 0 only)
      if (wr_ctrl && avs_byteenable[0]) begin
        irq_en <= avs_writedata[2];
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
            err_code <= ERR_START_BUSY;
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

  assign irq = irq_en & (done | error);

  // read mux, registered (latency 1)
  always_ff @(posedge clk) begin
    if (avs_read) begin
      case (avs_address)
        6'h00: avs_readdata <= VERSION_VALUE;
        6'h01: avs_readdata <= {29'b0, irq_en, 2'b0};
        6'h02: avs_readdata <= {29'b0, error, done, busy};
        6'h03: avs_readdata <= {24'b0, err_code};
        6'h04: avs_readdata <= op;
        6'h05: avs_readdata <= dim_m;
        6'h06: avs_readdata <= dim_n;
        6'h07: avs_readdata <= dim_k;
        6'h08: avs_readdata <= addr_a;
        6'h09: avs_readdata <= addr_w;
        6'h0A: avs_readdata <= addr_out;
        6'h0B: avs_readdata <= addr_bias;
        6'h0C: avs_readdata <= addr_scale;
        6'h0D: avs_readdata <= stride_a;
        6'h0E: avs_readdata <= cycles[31:0];
        6'h0F: avs_readdata <= cycles[63:32];
        6'h10: avs_readdata <= bytes_rd;
        6'h11: avs_readdata <= bytes_wr;
        6'h12: avs_readdata <= heartbeat;
        6'h13: avs_readdata <= stride_out;
        6'h14: avs_readdata <= scratch;
        default: avs_readdata <= 32'hDEAD_BEEF;
      endcase
    end
  end
endmodule
