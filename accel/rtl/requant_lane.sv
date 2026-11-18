// One lane of the epilogue: bit-exact requantization of golden.requantize.
//   y = sat( ((acc + bias) * M0 + 2^(shift-1)) >>> shift )
// acc/bias are INT32, M0 is a 31-bit unsigned multiplier (always in [2^30, 2^31)),
// shift is 1..62 (30..42 in the real layers). The product needs up to 63 bits.
// Saturation is symmetric: +-127 (INT8) or +-32767 (INT16).
// Fully pipelined, one value per cycle, fixed latency RQ_LATENCY (fpgate_pkg).
//
// The 32x31 product is split into four 16/17-bit partial products (each fits one
// DSP 18x18 multiplier) that are summed in registered steps, instead of leaving the
// decomposition to the tools: this is what makes 150 MHz reachable.
(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF" *)
module requant_lane #(
  // 1: one more stage inside the multiplier (m split in two byte halves, partial sums one cycle later): latency +1.
  // For instances whose multipliers are built from logic (swiglu_lane); the epilogue lanes keep 0.
  parameter bit EXTRA_STAGE = 1'b0
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                in_valid,
  input  logic signed [31:0]  acc,
  input  logic signed [31:0]  bias,
  input  logic        [30:0]  m0,
  input  logic        [5:0]   shift,
  input  logic                out16,     // 1: INT16 output, 0: INT8 output
  output logic                out_valid,
  output logic signed [15:0]  y
);
  // stage 1: bias add
  logic signed [31:0] accb_1;
  logic        [30:0] m0_1;
  logic        [5:0]  sh_1;
  logic               o16_1, v_1;
  // stage 2: partial products. a = ah*2^16 + al, m = mh*2^16 + ml
  //   a*m = ah*mh*2^32 + (ah*ml + al*mh)*2^16 + al*ml
  logic signed [31:0] p_hh_2;     // 16s x 16s
  logic signed [32:0] p_hl_2;     // 16s x 17s
  logic signed [32:0] p_lh_2;     // 17s x 16s
  logic signed [33:0] p_ll_2;     // 17s x 17s
  logic        [5:0]  sh_2;
  logic               o16_2, v_2;
  // stage 3: middle terms
  logic signed [33:0] mid_3;
  logic signed [33:0] p_ll_3;
  logic signed [31:0] p_hh_3;
  logic        [5:0]  sh_3;
  logic               o16_3, v_3;
  // stage 4: low part and high part with the rounding constant
  logic signed [63:0] lo_4, hi_4;
  logic        [5:0]  sh_4;
  logic               o16_4, v_4;
  // stage 5: full product plus rounding
  logic signed [63:0] rnd_5;
  logic        [5:0]  sh_5;
  logic               o16_5, v_5;
  // stage 6: shift
  logic signed [63:0] shr_6;
  logic               o16_6, v_6;

  // EXTRA_STAGE only: split products (stage 2) and their sums (stage 2x)
  logic signed [25:0] q_hh_lo, q_hl_lo, q_lh_lo, q_ll_lo;
  logic signed [25:0] q_hh_hi, q_hl_hi, q_lh_hi, q_ll_hi;
  logic signed [31:0] p_hh_x;
  logic signed [32:0] p_hl_x, p_lh_x;
  logic signed [33:0] p_ll_x;
  logic        [5:0]  sh_2x;
  logic               o16_2x, v_2x;
  wire signed [31:0] p_hh_s = EXTRA_STAGE ? p_hh_x : p_hh_2;
  wire signed [32:0] p_hl_s = EXTRA_STAGE ? p_hl_x : p_hl_2;
  wire signed [32:0] p_lh_s = EXTRA_STAGE ? p_lh_x : p_lh_2;
  wire signed [33:0] p_ll_s = EXTRA_STAGE ? p_ll_x : p_ll_2;
  wire        [5:0]  sh_s   = EXTRA_STAGE ? sh_2x : sh_2;
  wire               o16_s  = EXTRA_STAGE ? o16_2x : o16_2;
  wire               v_s    = EXTRA_STAGE ? v_2x : v_2;

  wire signed [15:0] ah = accb_1[31:16];
  wire signed [16:0] al = {1'b0, accb_1[15:0]};
  wire signed [15:0] mh = {1'b0, m0_1[30:16]};
  wire signed [16:0] ml = {1'b0, m0_1[15:0]};

  always_ff @(posedge clk) begin
    accb_1 <= acc + bias;
    m0_1   <= m0;
    sh_1   <= shift;
    o16_1  <= out16;

    sh_2   <= sh_1;
    o16_2  <= o16_1;
    if (EXTRA_STAGE) begin
      // each product = x * m_lo (m[7:0] unsigned) + (x * m_hi) << 8, summed in the next stage
      q_hh_lo <= ah * signed'({1'b0, mh[7:0]});   q_hh_hi <= ah * signed'({1'b0, mh[14:8]});
      q_hl_lo <= ah * signed'({1'b0, ml[7:0]});   q_hl_hi <= ah * signed'({1'b0, ml[15:8]});
      q_lh_lo <= al * signed'({1'b0, mh[7:0]});   q_lh_hi <= al * signed'({1'b0, mh[14:8]});
      q_ll_lo <= al * signed'({1'b0, ml[7:0]});   q_ll_hi <= al * signed'({1'b0, ml[15:8]});
      sh_2x   <= sh_2;
      o16_2x  <= o16_2;
    end else begin
      p_hh_2 <= ah * mh;
      p_hl_2 <= ah * ml;
      p_lh_2 <= al * mh;
      p_ll_2 <= al * ml;
    end
    if (EXTRA_STAGE) begin
      p_hh_x <= 32'(q_hh_lo) + (32'(q_hh_hi) <<< 8);
      p_hl_x <= 33'(q_hl_lo) + (33'(q_hl_hi) <<< 8);
      p_lh_x <= 33'(q_lh_lo) + (33'(q_lh_hi) <<< 8);
      p_ll_x <= 34'(q_ll_lo) + (34'(q_ll_hi) <<< 8);
    end

    mid_3  <= 34'(p_hl_s) + 34'(p_lh_s);
    p_ll_3 <= p_ll_s;
    p_hh_3 <= p_hh_s;
    sh_3   <= sh_s;
    o16_3  <= o16_s;

    lo_4   <= (64'(mid_3) <<< 16) + 64'(p_ll_3);
    hi_4   <= (64'(p_hh_3) <<< 32) + (64'sd1 <<< (sh_3 - 6'd1));
    sh_4   <= sh_3;
    o16_4  <= o16_3;

    rnd_5  <= lo_4 + hi_4;
    sh_5   <= sh_4;
    o16_5  <= o16_4;

    shr_6  <= rnd_5 >>> sh_5;
    o16_6  <= o16_5;

    y <= sat(shr_6, o16_6);
  end

  function automatic logic signed [15:0] sat(input logic signed [63:0] v, input logic wide);
    logic signed [63:0] qm;
    qm = wide ? 64'sd32767 : 64'sd127;
    if (v > qm)       return 16'(qm);
    else if (v < -qm) return 16'(-qm);
    else              return 16'(v);
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v_1 <= 1'b0; v_2 <= 1'b0; v_2x <= 1'b0; v_3 <= 1'b0; v_4 <= 1'b0; v_5 <= 1'b0; v_6 <= 1'b0; out_valid <= 1'b0;
    end else begin
      v_1 <= in_valid; v_2 <= v_1; v_2x <= v_2; v_3 <= v_s; v_4 <= v_3; v_5 <= v_4; v_6 <= v_5; out_valid <= v_6;
    end
  end
endmodule
