// Resource/timing probe for the complete engine (fpgate_accel_core): the register slave
// is the only way in (so parameters stay real signals), and a synthetic responder stands
// in for the DDR3: it answers read bursts with LFSR data and folds everything written
// into a register. Only for Quartus numbers; not part of the final design.
module engine_probe_top #(
  parameter int LANES = 4
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic [5:0]  avs_address,
  input  logic        avs_write,
  input  logic [31:0] avs_writedata,
  input  logic [3:0]  avs_byteenable,
  input  logic        avs_read,
  output logic [31:0] avs_readdata,
  output logic        irq,
  output logic [7:0]  led
);
  logic [31:0]  rd_address, wr_address, wr_byteenable;
  logic         rd_read, rd_waitrequest, rd_readdatavalid, wr_write, wr_waitrequest;
  logic [7:0]   rd_burstcount, wr_burstcount;
  logic [255:0] rd_readdata, wr_writedata;

  fpgate_accel_core #(.LANES(LANES)) u_core (.*);

  // read responder
  logic [255:0] lfsr;
  logic [7:0]   left;
  logic [1:0]   lat;
  logic         serving;
  assign rd_waitrequest = serving;
  assign rd_readdata    = lfsr;
  assign rd_readdatavalid = serving && (lat == 0);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      lfsr <= 256'd1; serving <= 1'b0; left <= '0; lat <= '0;
    end else begin
      lfsr <= {lfsr[254:0], lfsr[255] ^ lfsr[253] ^ lfsr[250] ^ lfsr[245]};
      if (!serving && rd_read) begin
        serving <= 1'b1; left <= rd_burstcount; lat <= 2'd2;
      end else if (serving) begin
        if (lat != 0) lat <= lat - 1'b1;
        else begin
          left <= left - 1'b1;
          if (left == 1) serving <= 1'b0;
        end
      end
    end
  end

  // write sink
  assign wr_waitrequest = 1'b0;
  logic [255:0] fold;
  always_ff @(posedge clk) begin
    if (wr_write) fold <= fold ^ wr_writedata ^ {224'b0, wr_address};
    led <= fold[7:0] ^ fold[15:8] ^ fold[23:16] ^ fold[31:24];
  end
endmodule
