// One simple dual-port RAM of the softmax unit: 128-bit words (8 columns of 16-bit probabilities), one row
// of P. Written in pass 3, read out transposed. A module of its own with the standard template and the
// ramstyle attribute, so that Quartus infers M10K blocks (16 of these as registers would not fit the device).
module softmax_rowram #(
  parameter int DEPTH = 130,
  parameter int AW    = 8
)(
  input  logic           clk,
  input  logic           we,
  input  logic [AW-1:0]  waddr,
  input  logic [127:0]   wdata,
  input  logic [AW-1:0]  raddr,
  output logic [127:0]   q
);
  (* ramstyle = "M10K, no_rw_check" *) logic [127:0] mem [DEPTH];

  // the write side arrives from a lane output with a large fan-out: two register stages
  logic          we_1, we_2;
  logic [AW-1:0] wa_1, wa_2;
  logic [127:0]  wd_1, wd_2;
  always_ff @(posedge clk) begin
    we_1 <= we;   wa_1 <= waddr; wd_1 <= wdata;
    we_2 <= we_1; wa_2 <= wa_1;  wd_2 <= wd_1;
    if (we_2) mem[wa_2] <= wd_2;
    q <= mem[raddr];
  end
endmodule
