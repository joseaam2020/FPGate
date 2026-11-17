// Tile sequencer: walks one operation (M x N x K) over panels of subtiles and tiles
// of 16 output channels. It does not touch data; it issues DMA read commands,
// switches the weight ping-pong, starts the operand feeder per subtile, tells the
// output side where the rows of a subtile go, and watches for stalls.
//
//   for panel (<= sub_per_panel subtiles of 16 rows):
//     load the A panel; load W tile 0
//     for tile (16 output channels):
//       prefetch W tile+1 (other buffer) while computing
//       for subtile in panel: feed the array (its rows drain while the next one is fed)
//       wait for the prefetch, swap the weight buffers
//
// Subtiles are fed back to back; the array's shadow accumulators let a subtile drain
// while the next one computes. The output FIFO is protected row by row (the epilogue
// only takes a row when the engine has room for it). Buffers are only overwritten
// once no row that needs them is still in flight: the prefetch into the other
// weight buffer (which also holds the epilogue constants of the previous tile) waits
// for the subtiles of that tile to come out, a new panel's weight load waits for the
// datapath to be empty.
//
// Error codes: 2 bad parameters, 3 DMA read timeout, 4 DMA write timeout,
// 5 output FIFO overflow, 6 watchdog (no progress).
module tile_sequencer #(
  parameter int A_WORDS  = 16384,     // 128-bit words in the A RAM
  parameter int W_BEATS  = 1024,      // 256-bit entries per weight buffer
  parameter int WD_LIMIT = 1 << 20    // cycles without progress before error 6
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  // operation
  input  logic        op_in16,
  input  logic        op_out16,
  input  logic        op_raw32,       // raw INT32 accumulators: 64-byte output tile rows
  input  logic        op_tr,          // transposed output blocks (OUTFMT 1, 2): 16-bit outputs are not allowed
  input  logic        op_rope,        // RoPE in the epilogue: 16-bit and raw outputs are not allowed
  input  logic [31:0] out_tstride,    // bytes between consecutive output tiles (0: 16 / 32 / 64 by output type)
  input  logic [31:0] head_tiles,     // output tiles per head (0: no head split)
  input  logic [31:0] head_stride,    // bytes between the first tiles of consecutive heads
  input  logic [31:0] row0,           // sequence index of row 0 of the operation (RoPE token)
  input  logic [7:0]  sub_per_panel,
  input  logic [31:0] dim_m, dim_n, dim_k,
  input  logic [31:0] addr_a, addr_w, addr_out, stride_out,
  // read DMA
  output logic        rd_cmd_valid,
  input  logic        rd_cmd_ready,
  output logic [31:0] rd_cmd_addr,
  output logic [15:0] rd_cmd_beats,
  output logic        rd_to_a,        // 1: A panel, 0: weight tile
  input  logic        rd_done,
  input  logic        rd_err,
  // weight buffers
  output logic        w_load_sel,
  output logic        w_comp_sel,
  // feeder
  output logic        fd_start,
  output logic [15:0] fd_a_base,
  output logic [15:0] fd_k,
  output logic        fd_two_pass,
  // output side context (pulse together with fd_start)
  output logic        sub_start,
  output logic [31:0] sub_addr,
  output logic [4:0]  sub_rows_valid,
  output logic [15:0] sub_row0,       // sequence index of row 0 of the subtile
  output logic [1:0]  sub_th,         // output tile within its head (modulo 4)
  // events from the datapath
  input  logic        row_last_seen,  // the last row of a subtile has come out (one pulse per subtile)
  input  logic        lt_busy,
  input  logic        fd_busy,        // the feeder is walking a subtile
  input  logic        ctx_full,       // the output side holds two subtile contexts already
  input  logic        wr_idle,
  input  logic [7:0]  fifo_free,      // free entries of the output FIFO (registered)
  input  logic        progress,
  input  logic        ext_err_valid,
  input  logic [7:0]  ext_err_code,
  // status
  output logic        busy,
  output logic        done,
  output logic        err_valid,
  output logic [7:0]  err_code,
  output logic        abort
);
  typedef enum logic [4:0] {
    S_IDLE, S_CHK, S_CHK_LOOP, S_PSTART, S_ACALC, S_LOADA_CMD, S_LOADA_WAIT,
    S_LOADW_CMD, S_LOADW_WAIT, S_TSTART, S_PF_CMD, S_SSTART, S_RUN, S_DRAIN,
    S_TEND, S_PEND, S_FLUSH, S_DONE, S_ERR
  } state_t;
  state_t state, state_q;

  // latched operation
  logic        in16, out16, op_raw32_q;
  logic [7:0]  spp;
  logic [31:0] m, n;
  logic [15:0] k, kb, kw, planes_k;     // K, K/2, planes*K words
  logic [15:0] planes_kb;               // planes*K/2 beats per subtile
  logic [31:0] a_addr_cur, w_base, out_base, stride16, w_tile_bytes, out_tile_bytes;
  logic [31:0] head_tiles_q, head_stride_q, th, head_base;
  logic [15:0] row_base;

  // loop state
  logic [31:0] subs_left, nt, ct;
  logic [7:0]  panel_sub, sub_idx;
  logic [15:0] a_beats, a_base, chk_sum;
  logic [8:0]  ccnt;
  logic [31:0] panel_out_base, ct_base, panel_acc, w_next_addr;
  logic        pf_pending, pf_done;
  logic        pf_req;                  // prefetch wanted, not issued yet
  logic [2:0]  sub_out;                 // subtiles started whose last row is not out yet
  logic [2:0]  pf_wait;                 // of those, the ones of the previous tile
  logic [7:0]  next_err;

  // watchdog
  logic [31:0] wd;
  logic [7:0]  fifo_free_q;

  // the prefetch is issued from the compute states once the previous tile is out
  wire pf_state = (state == S_SSTART) || (state == S_RUN) || (state == S_DRAIN) || (state == S_TEND);
  wire pf_issue = pf_state && pf_req && (pf_wait == 3'd0);

  assign busy        = (state != S_IDLE);
  assign fd_k        = k;
  assign fd_two_pass = in16;
  assign fd_a_base   = a_base;

  wire is_last_sub = (subs_left - 32'(sub_idx) == 32'd1);

  always_comb begin
    rd_cmd_valid = 1'b0;
    rd_cmd_addr  = '0;
    rd_cmd_beats = '0;
    rd_to_a      = 1'b0;
    case (state)
      S_LOADA_CMD: begin
        rd_cmd_valid = 1'b1; rd_to_a = 1'b1;
        rd_cmd_addr = a_addr_cur; rd_cmd_beats = a_beats;
      end
      S_LOADW_CMD: begin
        // buffer 0 may still hold the constants of rows in flight
        rd_cmd_valid = !lt_busy && (sub_out == 3'd0);
        rd_cmd_addr = w_base; rd_cmd_beats = 16'd8 + kb;
      end
      default: ;
    endcase
    if (pf_issue) begin
      rd_cmd_valid = 1'b1;
      rd_cmd_addr = w_next_addr; rd_cmd_beats = 16'd8 + kb;
    end
  end

  always_ff @(posedge clk) begin
    done      <= 1'b0;
    err_valid <= 1'b0;
    abort     <= 1'b0;
    fd_start  <= 1'b0;
    sub_start <= 1'b0;
    state_q   <= state;
    fifo_free_q <= fifo_free;

    if (!rst_n) begin
      state <= S_IDLE;
      wd    <= '0;
    end else begin
      // the prefetch may finish while a subtile is computing: remember it
      if (rd_done && pf_pending) pf_done <= 1'b1;

      // subtiles in flight (started, last row not out yet)
      if (state == S_IDLE) sub_out <= '0;
      else                 sub_out <= sub_out + 3'(sub_start) - 3'(row_last_seen);
      if (row_last_seen && pf_wait != 3'd0) pf_wait <= pf_wait - 1'b1;
      if (pf_issue && rd_cmd_ready) begin
        pf_req      <= 1'b0;
        pf_pending  <= 1'b1;
        pf_done     <= 1'b0;
        w_next_addr <= w_next_addr + w_tile_bytes;
      end

      // watchdog
      if (state == S_IDLE || progress || state != state_q || fifo_free != fifo_free_q) wd <= '0;
      else                                                  wd <= wd + 1'b1;

      case (state)
        S_IDLE: if (start) begin
          in16 <= op_in16; out16 <= op_out16; op_raw32_q <= op_raw32; spp <= sub_per_panel;
          m <= dim_m; n <= dim_n; k <= dim_k[15:0];
          kb <= dim_k[15:1];
          kw <= dim_k[15:0];
          planes_k  <= op_in16 ? {dim_k[14:0], 1'b0} : dim_k[15:0];
          planes_kb <= op_in16 ? {1'b0, dim_k[15:1], 1'b0} : {1'b0, dim_k[15:1]};
          a_addr_cur <= addr_a; w_base <= addr_w; out_base <= addr_out;
          stride16 <= {stride_out[27:0], 4'b0};
          w_tile_bytes   <= {12'b0, dim_k[15:0], 4'b0} + 32'd256;
          out_tile_bytes <= (out_tstride != 0) ? out_tstride : op_raw32 ? 32'd64 : (op_out16 ? 32'd32 : 32'd16);
          head_tiles_q   <= head_tiles;
          head_stride_q  <= head_stride;
          row_base       <= row0[15:0];
          state <= S_CHK;
        end

        S_CHK: begin
          if (m == 0 || n == 0 || n[3:0] != 0 || k == 0 || k[0] ||
              (kb > 16'(W_BEATS)) || spp == 0 ||
              ((op_tr || op_rope) && (out16 || op_raw32_q))) begin
            next_err <= 8'd2; state <= S_ERR;
          end else begin
            chk_sum <= '0;
            ccnt    <= {1'b0, spp};
            state   <= S_CHK_LOOP;
          end
        end

        // does one panel (spp subtiles) fit in the A RAM?  sum = spp * planes*K
        S_CHK_LOOP: begin
          if (ccnt != 0) begin
            chk_sum <= chk_sum + planes_k;
            ccnt    <= ccnt - 1'b1;
          end else if (chk_sum > 16'(A_WORDS) || chk_sum < planes_k) begin
            next_err <= 8'd2; state <= S_ERR;
          end else begin
            subs_left      <= (m + 32'd15) >> 4;
            nt             <= n >> 4;
            panel_out_base <= out_base;
            state          <= S_PSTART;
          end
        end

        S_PSTART: begin
          panel_sub  <= (subs_left < 32'(spp)) ? subs_left[7:0] : spp;
          ccnt       <= {1'b0, (subs_left < 32'(spp)) ? subs_left[7:0] : spp};
          a_beats    <= '0;
          panel_acc  <= '0;
          state      <= S_ACALC;
        end

        S_ACALC: begin
          if (ccnt != 0) begin
            a_beats <= a_beats + planes_kb;
            ccnt    <= ccnt - 1'b1;
          end else begin
            state <= S_LOADA_CMD;
          end
        end

        S_LOADA_CMD: if (rd_cmd_ready) state <= S_LOADA_WAIT;

        S_LOADA_WAIT: if (rd_done) begin
          a_addr_cur <= a_addr_cur + {11'b0, a_beats, 5'b0};
          w_load_sel <= 1'b0;
          state      <= S_LOADW_CMD;
        end

        S_LOADW_CMD: if (rd_cmd_valid && rd_cmd_ready) state <= S_LOADW_WAIT;

        S_LOADW_WAIT: if (rd_done) begin
          w_comp_sel  <= 1'b0;
          w_load_sel  <= 1'b1;
          ct          <= '0;
          th          <= '0;
          head_base   <= panel_out_base;
          ct_base     <= panel_out_base;
          w_next_addr <= w_base + w_tile_bytes;
          state       <= S_TSTART;
        end

        S_TSTART: begin
          sub_idx    <= '0;
          sub_addr   <= ct_base;
          a_base     <= '0;
          pf_pending <= 1'b0;
          pf_done    <= 1'b0;
          // the other buffer still holds the constants of the subtiles in flight
          pf_req     <= (ct + 32'd1 < nt);
          pf_wait    <= sub_out - 3'(row_last_seen);
          state      <= S_SSTART;
        end

        // the output side keeps the address of two subtiles at most
        S_SSTART: if (!ctx_full) begin
          sub_rows_valid <= is_last_sub ? (5'(m - 32'd1) & 5'h0F) + 5'd1 : 5'd16;
          sub_row0       <= row_base + {4'b0, sub_idx, 4'b0};
          sub_th         <= th[1:0];
          sub_start      <= 1'b1;
          fd_start       <= 1'b1;
          state          <= S_RUN;
        end

        // the feeder takes the start one cycle late; its rows drain on their own
        S_RUN: if (state_q == S_RUN && !fd_busy) state <= S_DRAIN;

        S_DRAIN: begin
          a_base   <= a_base + planes_k;
          sub_addr <= sub_addr + stride16;
          if (ct == 0) panel_acc <= panel_acc + stride16;
          sub_idx  <= sub_idx + 1'b1;
          state    <= (sub_idx + 8'd1 < panel_sub) ? S_SSTART : S_TEND;
        end

        S_TEND: begin
          // wait for the prefetch (it may have finished long ago)
          if (!pf_req && (!pf_pending || pf_done)) begin
            pf_pending <= 1'b0;
            pf_done    <= 1'b0;
            w_comp_sel <= w_load_sel;
            w_load_sel <= w_comp_sel;
            ct         <= ct + 1'b1;
            if (head_tiles_q != 0 && th + 32'd1 == head_tiles_q) begin
              th        <= '0;
              head_base <= head_base + head_stride_q;
              ct_base   <= head_base + head_stride_q;
            end else begin
              th        <= th + 1'b1;
              ct_base   <= ct_base + out_tile_bytes;
            end
            state      <= (ct + 32'd1 < nt) ? S_TSTART : S_PEND;
          end
        end

        S_PEND: begin
          subs_left      <= subs_left - 32'(panel_sub);
          row_base       <= row_base + {4'b0, panel_sub, 4'b0};
          panel_out_base <= panel_out_base + panel_acc;
          state          <= (subs_left == 32'(panel_sub)) ? S_FLUSH : S_PSTART;
        end

        S_FLUSH: if (wr_idle && !lt_busy && sub_out == 3'd0) state <= S_DONE;

        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        S_ERR: begin
          err_valid <= 1'b1;
          err_code  <= next_err;
          abort     <= 1'b1;
          state     <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase

      // asynchronous error sources (after the case: the last assignment wins)
      if (state != S_IDLE && state != S_ERR) begin
        if (rd_err)                   begin next_err <= 8'd3; state <= S_ERR; end
        else if (ext_err_valid)       begin next_err <= ext_err_code; state <= S_ERR; end
        else if (wd >= 32'(WD_LIMIT)) begin next_err <= 8'd6; state <= S_ERR; end
      end
    end
  end
endmodule
