// Avalon-MM slave memory model for the DMA testbenches (256-bit data, byte addresses).
// Random wait states, random gaps and latency on read data, byte enables on writes.
// One read burst is served at a time (waitrequest stays high meanwhile); writes are
// single-beat. `hang_reads` makes the model swallow read requests without answering,
// to exercise the DMA timeout. Backing store: DEPTH beats starting at address 0.
module avalon_mem_model #(
  parameter int DEPTH    = 1 << 16,   // beats (32 bytes each)
  parameter int WAIT_PCT = 25,        // % of cycles with waitrequest high
  parameter int GAP_PCT  = 25,        // % of cycles without readdatavalid inside a burst
  parameter int LAT      = 4          // cycles between accepting a read and the first beat
)(
  input  logic         clk,
  input  logic         rst_n,
  input  logic [31:0]  address,
  input  logic         read,
  input  logic         write,
  input  logic [255:0] writedata,
  input  logic [31:0]  byteenable,
  input  logic [7:0]   burstcount,
  output logic         waitrequest,
  output logic [255:0] readdata,
  output logic         readdatavalid,
  input  logic         hang_reads
);
  logic [255:0] mem [DEPTH];

  int unsigned rd_beats_total, wr_beats_total;
  int unsigned bad_access;

  // read burst state
  logic        serving;
  logic [31:0] rd_addr;
  logic [7:0]  rd_left;
  int          lat_cnt;
  logic        rnd_wait;

  always @(posedge clk) rnd_wait <= (WAIT_PCT > 0) && ($urandom_range(99, 0) < WAIT_PCT);

  assign waitrequest = !rst_n || rnd_wait || serving;

  initial begin
    rd_beats_total = 0; wr_beats_total = 0; bad_access = 0;
    serving = 0; readdatavalid = 0; readdata = '0; rd_left = 0; rd_addr = 0; lat_cnt = 0;
  end

  always @(posedge clk) begin
    readdatavalid <= 1'b0;
    if (!rst_n) begin
      serving <= 1'b0;
    end else begin
      // accept a command
      if (!waitrequest && write) begin
        if (address[31:5] >= DEPTH || address[4:0] != 0 || burstcount != 1) bad_access++;
        else begin
          for (int b = 0; b < 32; b++)
            if (byteenable[b]) mem[address[31:5]][8*b +: 8] <= writedata[8*b +: 8];
          wr_beats_total++;
        end
      end
      if (!waitrequest && read) begin
        if (address[31:5] + burstcount > DEPTH || address[4:0] != 0 || burstcount == 0) bad_access++;
        else if (!hang_reads) begin
          serving <= 1'b1;
          rd_addr <= address[31:5];
          rd_left <= burstcount;
          lat_cnt <= LAT;
        end
      end
      // serve the burst
      if (serving) begin
        if (lat_cnt > 0) begin
          lat_cnt <= lat_cnt - 1;
        end else if ($urandom_range(99, 0) >= GAP_PCT) begin
          readdata      <= mem[rd_addr];
          readdatavalid <= 1'b1;
          rd_addr       <= rd_addr + 1'b1;
          rd_left       <= rd_left - 1'b1;
          rd_beats_total++;
          if (rd_left == 1) serving <= 1'b0;
        end
      end
    end
  end
endmodule
