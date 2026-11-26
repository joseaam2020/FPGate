// PE_ROWS x PE_COLS output-stationary array with skewed injection and
// row-serial drain.
//
// Interface contract (one k-step per cycle, unskewed):
//   a_vec[i] / w_vec[j] carry the k-th element of row i / column j.
//   in_first marks the first k-step of a tile, in_shift the first k-step of
//   the second (low-byte) pass, in_last the very last k-step of the tile.
// The array skews row i by i cycles and column j by j cycles, so PE(i,j)
// sees matching a/w pairs. Every PE copies its finished accumulator to a
// shadow register and the tile drains from the shadows, one row per cycle, on
// drain_* (valid/ready: the array holds the row while drain_ready is low), so
// the next tile can be fed right behind the last k-step of the previous one.
// The one rule for the producer: the last k-step (in_last) of a tile may only
// be presented while `last_ok` is high (all shadows of the previous tile have
// been read). Any other k-step, and bubbles (in_valid low), may be fed at any time.
// `busy` is high from the first in_valid until the last row of the last tile fed
// has been delivered. `in_tag` is sampled with in_last and comes back with the
// rows of that tile on `drain_tag`.
module gemm_array
  import fpgate_pkg::*;
#(
  parameter int ROWS = PE_ROWS,
  parameter int COLS = PE_COLS
)(
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              in_valid,
  input  logic                              in_first,
  input  logic                              in_shift,
  input  logic                              in_last,
  input  logic                              in_tag,
  input  logic signed [ROWS-1:0][ACT_W-1:0] a_vec,
  input  logic signed [COLS-1:0][WGT_W-1:0] w_vec,
  output logic                              busy,
  output logic                              last_ok,       // in_last may be presented (registered)
  output logic                              drain_tag,
  output logic                              drain_valid,
  input  logic                              drain_ready,   // consumer takes the row (valid/ready)
  output logic                              drain_last,    // with the last row of the tile
  output logic [$clog2(ROWS)-1:0]           drain_row,
  output logic signed [COLS-1:0][ACC_W-1:0] drain_data
);
  // PE(i,j) sees the last k-step i+j cycles after in_last and holds it in its shadow
  // from cycle i+j+3. Row r is read DLY+1+r cycles after in_last, and its last column
  // (j = COLS-1) needs r+COLS+2, so DLY = COLS+1 is the minimum; one cycle of margin.
  localparam int LAT = COLS + 2;

  genvar i, j;

  // ---- injection skew ----------------------------------------------------
  logic signed [ACT_W-1:0] a_sk   [ROWS];
  logic                    v_sk   [ROWS];
  logic                    f_sk   [ROWS];
  logic                    s_sk   [ROWS];
  logic                    l_sk   [ROWS];
  logic signed [WGT_W-1:0] w_sk   [COLS];

  generate
  for (i = 0; i < ROWS; i++) begin : g_askew
    if (i == 0) begin : g_direct
      assign a_sk[i] = a_vec[i];
      assign v_sk[i] = in_valid;
      assign f_sk[i] = in_first;
      assign s_sk[i] = in_shift;
      assign l_sk[i] = in_last;
    end else begin : g_delay
      logic signed [ACT_W-1:0] a_d [i];
      logic                    v_d [i];
      logic                    f_d [i];
      logic                    s_d [i];
      logic                    l_d [i];
      always_ff @(posedge clk) begin
        a_d[0] <= a_vec[i];
        f_d[0] <= in_first;
        s_d[0] <= in_shift;
        for (int d = 1; d < i; d++) begin
          a_d[d] <= a_d[d-1];
          f_d[d] <= f_d[d-1];
          s_d[d] <= s_d[d-1];
        end
      end
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          for (int d = 0; d < i; d++) begin v_d[d] <= 1'b0; l_d[d] <= 1'b0; end
        end else begin
          v_d[0] <= in_valid;
          l_d[0] <= in_valid & in_last;
          for (int d = 1; d < i; d++) begin v_d[d] <= v_d[d-1]; l_d[d] <= l_d[d-1]; end
        end
      end
      assign a_sk[i] = a_d[i-1];
      assign v_sk[i] = v_d[i-1];
      assign f_sk[i] = f_d[i-1];
      assign s_sk[i] = s_d[i-1];
      assign l_sk[i] = l_d[i-1];
    end
  end

  endgenerate

  generate
  for (j = 0; j < COLS; j++) begin : g_wskew
    if (j == 0) begin : g_direct
      assign w_sk[j] = w_vec[j];
    end else begin : g_delay
      logic signed [WGT_W-1:0] w_d [j];
      always_ff @(posedge clk) begin
        w_d[0] <= w_vec[j];
        for (int d = 1; d < j; d++) w_d[d] <= w_d[d-1];
      end
      assign w_sk[j] = w_d[j-1];
    end
  end
  endgenerate

  // ---- PE grid -----------------------------------------------------------
  logic signed [ACT_W-1:0] a_w  [ROWS][COLS+1];
  logic                    v_w  [ROWS][COLS+1];
  logic                    f_w  [ROWS][COLS+1];
  logic                    s_w  [ROWS][COLS+1];
  logic                    l_w  [ROWS][COLS+1];
  logic signed [WGT_W-1:0] w_w  [ROWS+1][COLS];
  logic signed [ACC_W-1:0] sh_w [ROWS][COLS];

  generate
  for (i = 0; i < ROWS; i++) begin : g_row
    assign a_w[i][0] = a_sk[i];
    assign v_w[i][0] = v_sk[i];
    assign f_w[i][0] = f_sk[i];
    assign s_w[i][0] = s_sk[i];
    assign l_w[i][0] = l_sk[i];
  end
  for (j = 0; j < COLS; j++) begin : g_col
    assign w_w[0][j] = w_sk[j];
  end

  for (i = 0; i < ROWS; i++) begin : g_pe_row
    for (j = 0; j < COLS; j++) begin : g_pe_col
      pe #(.ACC_W(ACC_W)) u_pe (
        .clk(clk), .rst_n(rst_n),
        .a_in(a_w[i][j]),         .a_valid_in(v_w[i][j]),
        .a_first_in(f_w[i][j]),   .a_shift_in(s_w[i][j]),   .a_last_in(l_w[i][j]),
        .a_out(a_w[i][j+1]),      .a_valid_out(v_w[i][j+1]),
        .a_first_out(f_w[i][j+1]), .a_shift_out(s_w[i][j+1]), .a_last_out(l_w[i][j+1]),
        .w_in(w_w[i][j]),         .w_out(w_w[i+1][j]),
        .shadow(sh_w[i][j])
      );
    end
  end
  endgenerate

  // ---- drain (from the shadows) ---------------------------------------------
  logic [LAT-1:0] last_pipe, tag_pipe;
  logic           start;
  logic           active, tag_act;
  logic [$clog2(ROWS)-1:0] cnt;
  // one copy of the row counter per column: the 16:1 row mux of 512 bits would otherwise
  // hang on a single high-fanout register
  (* dont_merge *) logic [$clog2(ROWS)-1:0] cnt_c [COLS];
  logic           feeding;              // between the first k-step of a tile and its in_last
  logic [1:0]     n_unread;             // tiles fed whose shadows are not all read yet
  logic [1:0]     n_out;                // tiles fed whose last row is not delivered yet

  wire fed_last  = in_valid & in_last;
  wire load      = active && (!drain_valid || drain_ready);
  wire read_done = load && (cnt == ROWS-1);
  wire out_done  = drain_valid && drain_ready && drain_last;

  always_ff @(posedge clk) begin
    if (!rst_n) last_pipe <= '0;
    else        last_pipe <= {last_pipe[LAT-2:0], fed_last};
    tag_pipe <= {tag_pipe[LAT-2:0], in_tag};
  end
  assign start = last_pipe[LAT-1];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      active      <= 1'b0;
      cnt         <= '0;
      for (int j = 0; j < COLS; j++) cnt_c[j] <= '0;
      drain_valid <= 1'b0;
      drain_last  <= 1'b0;
      drain_row   <= '0;
      feeding     <= 1'b0;
      n_unread    <= '0;
      n_out       <= '0;
      last_ok     <= 1'b1;
    end else begin
      if (in_valid) feeding <= !in_last;
      n_unread <= n_unread + 2'(fed_last) - 2'(read_done);
      n_out    <= n_out    + 2'(fed_last) - 2'(out_done);
      // registered: one cycle late, but only in_last itself can make it fall
      last_ok  <= (n_unread + 2'(fed_last) - 2'(read_done)) == 2'd0;
      // the output row register is released when the consumer takes it
      if (drain_valid && drain_ready) drain_valid <= 1'b0;
      if (start) begin
        // the producer's rule guarantees the previous tile has been read completely
        active  <= 1'b1;
        tag_act <= tag_pipe[LAT-1];
        cnt     <= '0;
        for (int j = 0; j < COLS; j++) cnt_c[j] <= '0;
      end else if (load) begin
        drain_valid <= 1'b1;
        drain_last  <= (cnt == ROWS-1);
        drain_row   <= cnt;
        drain_tag   <= tag_act;
        for (int j = 0; j < COLS; j++) drain_data[j] <= sh_w[cnt_c[j]][j];
        cnt <= cnt + 1'b1;
        for (int j = 0; j < COLS; j++) cnt_c[j] <= cnt + 1'b1;
        if (cnt == ROWS-1) active <= 1'b0;
      end
    end
  end
  assign busy = feeding | (n_out != 2'd0);
endmodule
