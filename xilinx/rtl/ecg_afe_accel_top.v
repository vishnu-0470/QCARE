// ============================================================================
//  ecg_afe_accel_top : streaming QRS detector + single-datapath beat engine
//  (window copy, AFE quality gate, amplitude delineation of QRS/P/T,
//   patient-referenced features, stage-1 decision tree with confidence early
//   exit, stage-2 4-bit/8-bit CNN on one shared MAC).
//  Bit-exact with matlab/{qrs_detect_golden,refine_R,beat_raw,beat_decide,
//  cnn_forward_int}.m.  Constants come from gen/params.vh (export_rtl.m),
//  weights from gen/wrom.mem and gen/tree.mem.
//  All memories use asynchronous reads (LUT-RAM); no BRAM, no DSP48 intended.
// ============================================================================
`timescale 1ns/1ps
module ecg_afe_accel_top #(
    parameter WROM_FILE = "gen/wrom.mem",
    parameter TREE_FILE = "gen/tree.mem"
)(
    input  wire        clk,
    input  wire        rst_n,        // hold low >= 64 cycles after power-up
    input  wire        s_valid,      // one pulse per ADC sample
    input  wire [11:0] s_data,
    output reg         out_valid,    // one pulse per classified beat
    output reg  [1:0]  out_cls,      // 0 N, 1 S, 2 V, 3 = dropped by AFE gate
    output reg         out_early,    // decided by stage-1 (no CNN)
    output reg         out_drop_sat, // drop caused by rail saturation
    output wire        busy,
    output reg         err_fifo,     // event FIFO overflow (sticky)
    output reg         err_ring      // window older than ring depth (sticky)
);
`include "params.vh"

    // ------------------------------------------------------------------ memories
    reg [11:0] ring  [0:1023];
    reg [12:0] eh    [0:31];
    reg [23:0] ev_fifo [0:15];
    reg [11:0] bwm   [0:359];
    reg signed [7:0] um   [0:119];
    reg [6:0]  y1m   [0:959];
    reg [6:0]  outm  [0:479];
    reg signed [7:0] vecm [0:39];
    reg [6:0]  hm    [0:15];
    reg [15:0] wrom  [0:1023];
    reg [47:0] trom  [0:255];
    initial begin
        $readmemh(WROM_FILE, wrom);
        $readmemh(TREE_FILE, trom);
    end

    // ================================================================ detector
    reg [23:0] wr_cnt;
    reg [11:0] x1, x2;
    reg [17:0] S;
    reg [15:0] spk, npk;
    reg        first_ev;
    reg [7:0]  rc;
    reg        dstate;
    reg [7:0]  tcnt;
    reg [15:0] mmax;
    reg [23:0] mpos;
    reg [5:0]  clr_cnt;
    reg [4:0]  wp, rp;
    wire [4:0] fifo_cnt = wp - rp;

    wire        first_s = (wr_cnt == 24'd0);
    wire [11:0] x2_eff  = first_s ? s_data : x2;
    wire signed [13:0] s_diff = $signed({2'b00, s_data}) - $signed({2'b00, x2_eff});
    wire signed [13:0] s_neg  = -s_diff;
    wire [12:0] e_abs  = s_diff[13] ? s_neg[12:0] : s_diff[12:0];
    wire [4:0]  ix     = wr_cnt[4:0];
    wire [12:0] e_old  = eh[ix];
    wire [17:0] S_new  = S + {5'b0, e_abs} - {5'b0, e_old};
    wire [15:0] m      = {3'b0, S_new[17:5]};
    wire [15:0] dd     = (spk > npk) ? (spk - npk) : 16'd0;
    wire [17:0] dd3    = {dd, 1'b0} + {2'b0, dd};
    wire [15:0] thr_a  = npk + {1'b0, dd3[17:3]};
    wire [15:0] thr    = (thr_a > DET_TMIN) ? thr_a : DET_TMIN;
    wire signed [16:0] dn   = $signed({1'b0, m}) - $signed({1'b0, npk});
    wire signed [16:0] dn_s = dn >>> DET_NPK_SH;
    wire signed [16:0] npk_n = $signed({1'b0, npk}) + dn_s;
    wire        gt_mmax = (m > mmax);
    wire [15:0] mmax_n  = gt_mmax ? m : mmax;
    wire [23:0] mpos_n  = gt_mmax ? wr_cnt : mpos;
    wire signed [16:0] ds   = $signed({1'b0, mmax_n}) - $signed({1'b0, spk});
    wire signed [16:0] ds_s = ds >>> DET_SPK_SH;
    wire signed [16:0] spk_n = $signed({1'b0, spk}) + ds_s;

    always @(posedge clk) begin
        if (!rst_n) begin
            wr_cnt <= 24'd0; x1 <= 12'd0; x2 <= 12'd0; S <= 18'd0; spk <= 16'd0; npk <= 16'd0;
            first_ev <= 1'b1; rc <= 8'd0; dstate <= 1'b0; tcnt <= 8'd0; mmax <= 16'd0; mpos <= 24'd0;
            wp <= 5'd0; err_fifo <= 1'b0;
            eh[clr_cnt[4:0]] <= 13'd0;
            clr_cnt <= clr_cnt + 6'd1;
        end else if (s_valid) begin
            x1 <= s_data; x2 <= first_s ? s_data : x1;
            eh[ix] <= e_abs; S <= S_new;
            ring[wr_cnt[9:0]] <= s_data;
            wr_cnt <= wr_cnt + 24'd1;
            if (rc != 8'd0) rc <= rc - 8'd1;
            if (!dstate) begin
                if (rc == 8'd0) begin
                    if (m > thr) begin
                        dstate <= 1'b1; tcnt <= 8'd0; mmax <= m; mpos <= wr_cnt; rc <= DET_REFR;
                    end else begin
                        npk <= npk_n[15:0];
                    end
                end
            end else begin
                tcnt <= tcnt + 8'd1;
                mmax <= mmax_n; mpos <= mpos_n;
                if ((tcnt + 8'd1) == DET_TRACK) begin
                    if (fifo_cnt == 5'd16) err_fifo <= 1'b1;
                    else begin ev_fifo[wp[3:0]] <= mpos_n; wp <= wp + 5'd1; end
                    spk <= first_ev ? mmax_n : spk_n[15:0];
                    first_ev <= 1'b0; dstate <= 1'b0;
                end
            end
        end
    end
    initial clr_cnt = 6'd0;

    // ================================================================ engine
    localparam [4:0]
        E_IDLE=0,  E_RF0=1,  E_RF=2,   E_CP0=3,  E_CP=4,   E_GATE=5, E_RAMP=6, E_ON=7,
        E_OFF=8,   E_P0=9,   E_P1=10,  E_P2=11,  E_P=12,   E_T0=13,  E_T=14,   E_DIV=15,
        E_DIVW=16, E_TREE=17,E_NRM=18, E_DS=19,   E_C1=20,  E_C2=21,  E_FQ=22,  E_POOL=23,
        E_FC1=24,  E_FC2=25, E_ARG=26, E_UPD=27,  E_OUT=28, E_POP=29;
    reg [4:0] es;
    assign busy = (es != E_IDLE);

    reg [23:0] c_cur, c_next, c_prev;
    reg        have_prev;
    reg [23:0] R;
    reg [11:0] ref0;
    reg [11:0] best12;           // R-refinement best
    reg [8:0]  idx;
    reg [16:0] sumB;
    reg [8:0]  sat_cnt;
    reg [11:0] vmin, vmax, base;
    reg [11:0] ramp, th;
    reg [8:0]  onset, offset, qrs_w, jj;
    reg [8:0]  ja, pw0, pw1, jp, tw1, jt;
    reg [14:0] e0, e1, m5b, b5;
    reg [17:0] best;
    reg [13:0] p_amp, t_amp;
    reg [5:0]  pr;
    reg        tpol;
    reg [8:0]  rt;
    reg [10:0] fv [0:9];
    reg [3:0]  dv_i;
    reg        dv_start;
    reg [7:0]  tnode;
    reg [3:0]  tstep;
    reg [1:0]  leaf_cls;
    reg [7:0]  leaf_pur;
    reg [1:0]  cls_r;
    reg        early_r, drop_r, dsat_r;
    reg [12:0] mx;
    reg [2:0]  sh;
    reg [8:0]  a3;
    reg [3:0]  oc;
    reg [6:0]  ci;
    reg [1:0]  kk;
    reg [2:0]  ic;
    reg signed [23:0] acc;
    reg [3:0]  pc;
    reg [6:0]  pi;
    reg [6:0]  pA, mQ;
    reg [11:0] sP, sT;
    reg [4:0]  fj;
    reg [5:0]  fi;
    reg [9:0]  wfi;
    reg signed [23:0] lg0, lg1, lg2;
    reg [15:0] rr_ref, qrs_ref, p_ref, t_ref, r_ref, rt_ref;
    reg signed [4:0] pol_cnt;
    reg [7:0]  n_rr, n_mor;
    // temporaries (blocking, combinational use only)
    reg [17:0] best_n; reg [8:0] jp_n, jt_n; reg [14:0] m5b_n;
    reg [12:0] mxn; reg signed [23:0] acc_n; reg signed [23:0] dev_t; reg signed [23:0] fq_t;
    reg [6:0]  pA_n, mQ_n; reg [11:0] sP_n, sT_n; reg [1:0] lc_n; reg [7:0] lp_n;

    // ---- fifo head access
    wire [23:0] c_head  = ev_fifo[rp[3:0]];
    wire [23:0] c_head2 = ev_fifo[rp[3:0] + 4'd1];
    wire        head_win_ok = (c_head < ENG_MIN_C) || (wr_cnt >= c_head + ENG_READY);
    wire        head_to     = (wr_cnt >= c_head + WAIT_T);
    wire        start_ok    = head_win_ok && ((fifo_cnt >= 5'd2) || (fifo_cnt == 5'd1 && head_to));

    // ---- ring / window reads
    wire [9:0]  rf_ref_a = c_cur[9:0] - REF_BACK;
    wire [23:0] rf_i     = c_cur + {15'd0, idx} - SRCH_LO;
    wire [23:0] cp_a     = R + {15'd0, idx} - 24'd140;
    wire [9:0]  ring_ra  = (es == E_RF0) ? rf_ref_a : ((es == E_RF) ? rf_i[9:0] : cp_a[9:0]);
    wire [11:0] ring_rd  = ring[ring_ra];
    wire [11:0] rf_v     = ring_rd;
    wire [11:0] rf_d     = (rf_v > ref0) ? (rf_v - ref0) : (ref0 - rf_v);
    wire [11:0] cp_v     = ring_rd;

    // ---- window reads for delineation
    function [11:0] absd;                       // |a-b|
        input [11:0] a; input [11:0] b;
        begin absd = (a > b) ? (a - b) : (b - a); end
    endfunction
    wire [11:0] ad_j   = absd(bwm[jj], base);
    wire [11:0] ad_jp1 = absd(bwm[jj + 9'd1], base);
    wire [11:0] ad_jm1 = absd(bwm[jj - 9'd1], base);
    wire [14:0] m5v = bwm[ja - 9'd2] + bwm[ja - 9'd1] + bwm[ja] + bwm[ja + 9'd1] + bwm[ja + 9'd2];
    wire [14:0] base5 = {base, 2'b00} + {3'b0, base};

    // ---- divider
    wire [23:0] rr_post_w = c_next - c_cur;
    wire [15:0] rr_post_c = (rr_post_w > RR_MAX) ? RR_MAX : rr_post_w[15:0];
    wire [23:0] rr_pre_w  = have_prev ? (c_cur - c_prev) : {8'd0, rr_post_c};
    wire [15:0] rr_pre_c  = (rr_pre_w > RR_MAX) ? RR_MAX : rr_pre_w[15:0];
    reg  [15:0] dv_x, dv_ref;
    always @* begin
        case (dv_i)
            4'd0: begin dv_x = rr_pre_c;  dv_ref = rr_ref;  end
            4'd1: begin dv_x = rr_post_c; dv_ref = rr_ref;  end
            4'd2: begin dv_x = {7'd0, qrs_w}; dv_ref = qrs_ref; end
            4'd3: begin dv_x = {2'd0, p_amp}; dv_ref = p_ref;   end
            4'd4: begin dv_x = {2'd0, t_amp}; dv_ref = t_ref;   end
            4'd5: begin dv_x = {4'd0, ramp};  dv_ref = r_ref;   end
            default: begin dv_x = {7'd0, rt}; dv_ref = rt_ref;  end
        endcase
    end
    wire [15:0] dv_eff = (dv_ref == 16'd0) ? ((dv_x == 16'd0) ? 16'd1 : dv_x) : dv_ref;
    wire        dv_done;
    wire [23:0] dv_q;
    ecg_div24 u_div (.clk(clk), .rst_n(rst_n), .start(dv_start), .num({dv_x, 8'd0}),
                     .den({8'd0, dv_eff}), .done(dv_done), .q(dv_q));
    wire [10:0] dv_fv = (dv_q > 24'd1023) ? 11'd1023 : dv_q[10:0];

    // ---- tree
    wire [47:0] tw      = trom[tnode];
    wire        t_leaf  = tw[41];
    wire [3:0]  t_feat  = tw[40:37];
    wire [10:0] t_thr   = tw[36:26];
    wire [7:0]  t_left  = tw[25:18];
    wire [7:0]  t_right = tw[17:10];
    wire [1:0]  t_cls   = tw[9:8];
    wire [7:0]  t_pur   = tw[7:0];
    wire [10:0] t_fval  = fv[t_feat];

    // ---- CNN datapath helpers
    function [2:0] shamt;
        input [12:0] v;
        begin
            if (v <= 13'd127) shamt = 3'd0;
            else if (v <= 13'd255) shamt = 3'd1;
            else if (v <= 13'd511) shamt = 3'd2;
            else if (v <= 13'd1023) shamt = 3'd3;
            else if (v <= 13'd2047) shamt = 3'd4;
            else shamt = 3'd5;
        end
    endfunction
    function signed [7:0] wvq;                  // clamp((bw-base) >>> sh, -128, 127)
        input [11:0] bw; input [11:0] bs; input [2:0] s;
        reg signed [12:0] d; reg signed [12:0] dsh;
        begin
            d = $signed({1'b0, bw}) - $signed({1'b0, bs});
            dsh = d >>> s;
            if (dsh > 13'sd127) wvq = 8'sd127;
            else if (dsh < -13'sd128) wvq = -8'sd128;
            else wvq = dsh[7:0];
        end
    endfunction
    function [6:0] clamp127;                    // ReLU + clamp to 0..127
        input signed [23:0] v;
        begin
            if (v < 0) clamp127 = 7'd0;
            else if (v > 24'sd127) clamp127 = 7'd127;
            else clamp127 = v[6:0];
        end
    endfunction

    wire signed [7:0] wv0 = wvq(bwm[a3],           base, sh);
    wire signed [7:0] wv1 = wvq(bwm[a3 + 9'd1],    base, sh);
    wire signed [7:0] wv2 = wvq(bwm[a3 + 9'd2],    base, sh);
    wire signed [9:0] wsum = $signed({{2{wv0[7]}}, wv0}) + $signed({{2{wv1[7]}}, wv1}) + $signed({{2{wv2[7]}}, wv2});
    wire signed [9:0] wsum4 = wsum >>> 2;

    // conv1 operand
    wire [7:0]  c1_p   = {1'b0, ci} + {6'd0, kk};                 // i+k, valid 1..120
    wire        c1_ok  = (c1_p != 8'd0) && (c1_p != 8'd121);
    wire signed [7:0] c1_act = c1_ok ? um[c1_p - 8'd1] : 8'sd0;
    wire [9:0]  oc3    = {5'd0, oc, 1'b0} + {6'd0, oc};                          // oc*3
    wire [9:0]  c1_wa  = ROM_W1 + oc3 + {8'd0, kk};
    wire [9:0]  oc120  = {6'd0, oc} * 10'd120;
    wire [9:0]  c1_ya  = oc120 + {3'd0, ci};
    // conv2 operand
    wire [7:0]  c2_p   = {1'b0, ci} + {6'd0, kk};
    wire        c2_ok  = (c2_p != 8'd0) && (c2_p != 8'd121);
    wire [9:0]  c2_ya  = {7'd0, ic} * 10'd120 + {2'd0, c2_p} - 10'd1;
    wire [9:0]  pl_a   = {6'd0, pc} * 10'd120 + {3'd0, pi};
    wire [9:0]  y1_ra   = (es == E_POOL) ? pl_a : c2_ya;
    wire [6:0]  y1_rd   = y1m[y1_ra];
    wire [6:0]  c2_act7 = c2_ok ? y1_rd : 7'd0;
    wire [9:0]  c2_wa  = ROM_W2 + {6'd0, oc} * 10'd12 + {7'd0, ic} * 10'd3 + {8'd0, kk};
    wire [9:0]  c2_ra  = oc120 + {3'd0, ci};
    // pooling operand
    wire [6:0]  pl_o   = (pc < 4'd4) ? outm[pl_a] : y1_rd;

    // MAC (shared by conv1, conv2, fc1, fc2).  ONE ROM read port: every output
    // first spends a cycle loading its bias (bp=1), then runs its MAC cycles.
    reg                bp;
    reg  signed [7:0]  mac_a;      // activation
    reg  [9:0]         mac_wa;     // ROM address (weight or bias)
    always @* begin
        mac_a = 8'sd0; mac_wa = 10'd0;
        case (es)
            E_C1:  begin mac_a = c1_act;            mac_wa = bp ? (ROM_B1  + {6'd0, oc})    : c1_wa; end
            E_C2:  begin mac_a = {1'b0, c2_act7};   mac_wa = bp ? (ROM_B2  + {6'd0, oc})    : c2_wa; end
            E_FC1: begin mac_a = vecm[fi[5:0]];     mac_wa = bp ? (ROM_BF1 + {5'd0, fj})    : wfi;   end
            E_FC2: begin mac_a = {1'b0, hm[fi[3:0]]}; mac_wa = bp ? (ROM_BF2 + {8'd0, fj[1:0]}) : wfi; end
            default: ;
        endcase
    end
    wire [15:0]        wword = wrom[mac_wa];
    wire [3:0]         wn    = wword[3:0];
    wire signed [19:0] a20   = {{12{mac_a[7]}}, mac_a};
    wire signed [19:0] prod  = (wn[0] ? a20 : 20'sd0) + (wn[1] ? (a20 <<< 1) : 20'sd0)
                             + (wn[2] ? (a20 <<< 2) : 20'sd0) - (wn[3] ? (a20 <<< 3) : 20'sd0);
    wire signed [23:0] bias_s = {{8{wword[15]}}, wword};

    wire [10:0] fv6 = fv[6];
    // feature -> int8 for the CNN
    reg signed [7:0] fq_v;
    always @* begin
        fq_v = 8'sd0;
        if (idx[2:0] == 3'd6) fq_v = fv6[0] ? 8'sd64 : 8'sd0;
        else begin
            fq_t = $signed({13'd0, fv[idx[2:0]]}) - 24'sd256;
            fq_t = fq_t >>> 2;
            if (fq_t > 24'sd127) fq_v = 8'sd127;
            else if (fq_t < -24'sd128) fq_v = -8'sd128;
            else fq_v = fq_t[7:0];
        end
    end

    // stochastic-median reference update
    function [15:0] refnext;
        input [15:0] rf; input [15:0] x; input [7:0] nupd;
        reg [15:0] stp; reg [3:0] s;
        begin
            if (rf == 16'd0) refnext = (x == 16'd0) ? 16'd1 : x;
            else begin
                s = (nupd < REF_WARM_N) ? REF_SH_WARM : REF_SH_RUN;
                stp = rf >> s; if (stp == 16'd0) stp = 16'd1;
                if (x > rf) refnext = rf + stp;
                else if (x < rf) refnext = (rf > stp) ? (rf - stp) : 16'd1;
                else refnext = rf;
            end
        end
    endfunction


    always @(posedge clk) begin
        out_valid <= 1'b0; dv_start <= 1'b0;
        if (!rst_n) begin
            es <= E_IDLE; rp <= 5'd0; have_prev <= 1'b0; c_prev <= 24'd0; err_ring <= 1'b0;
            rr_ref <= 16'd0; qrs_ref <= 16'd0; p_ref <= 16'd0; t_ref <= 16'd0; r_ref <= 16'd0; rt_ref <= 16'd0;
            pol_cnt <= 5'sd0; n_rr <= 8'd0; n_mor <= 8'd0;
            out_cls <= 2'd0; out_early <= 1'b0; out_drop_sat <= 1'b0; bp <= 1'b0;
        end else begin
            case (es)
            E_IDLE: if (start_ok) begin
                        c_cur <= c_head; c_next <= (fifo_cnt >= 5'd2) ? c_head2 : (c_head + RR_MAX);
                        if ((wr_cnt - c_head) > 24'd830 && c_head >= ENG_MIN_C) err_ring <= 1'b1;
                        es <= (c_head < ENG_MIN_C) ? E_POP : E_RF0;
                    end
            E_RF0: begin
                        ref0 <= ring_rd; best12 <= 12'd0; R <= c_cur - SRCH_LO; idx <= 9'd0; es <= E_RF;
                    end
            E_RF:  begin
                        if (rf_d > best12) begin best12 <= rf_d; R <= rf_i; end
                        if (idx == 9'd38) es <= E_CP0; else idx <= idx + 9'd1;
                    end
            E_CP0: begin idx <= 9'd0; sumB <= 17'd0; sat_cnt <= 9'd0; vmin <= 12'd4095; vmax <= 12'd0; es <= E_CP; end
            E_CP:  begin
                        bwm[idx] <= cp_v;
                        if (idx >= BASE_LO && idx <= BASE_HI) sumB <= sumB + {5'd0, cp_v};
                        if (idx >= SAT_LO && idx <= SAT_HI && (cp_v >= 12'd4094 || cp_v <= 12'd1)) sat_cnt <= sat_cnt + 9'd1;
                        if (cp_v < vmin) vmin <= cp_v;
                        if (cp_v > vmax) vmax <= cp_v;
                        if (idx == 9'd359) es <= E_GATE; else idx <= idx + 9'd1;
                    end
            E_GATE: begin
                        base <= sumB[16:5];
                        dsat_r <= (sat_cnt >= SAT_TH);
                        if (sat_cnt >= SAT_TH || (vmax - vmin) < FLAT_TH) begin
                            cls_r <= 2'd3; early_r <= 1'b0; drop_r <= 1'b1; es <= E_OUT;
                        end else begin drop_r <= 1'b0; es <= E_RAMP; end
                    end
            E_RAMP: begin
                        // |bw[140]-base|, floor of 1
                        if (absd(bwm[140], base) == 12'd0) begin ramp <= 12'd1; th <= 12'd4; end
                        else begin
                            ramp <= absd(bwm[140], base);
                            th <= ((absd(bwm[140], base) >> W_SH) < 12'd4) ? 12'd4 : (absd(bwm[140], base) >> W_SH);
                        end
                        jj <= ON_START; es <= E_ON;
                    end
            E_ON:  begin
                        if (ad_j < th && ad_jp1 < th) begin onset <= jj; jj <= OFF_START; es <= E_OFF; end
                        else if (jj == ON_MIN) begin onset <= ON_MIN; jj <= OFF_START; es <= E_OFF; end
                        else jj <= jj - 9'd1;
                    end
            E_OFF: begin
                        if (ad_j < th && ad_jm1 < th) begin offset <= jj; qrs_w <= jj - onset; es <= E_P0; end
                        else if (jj == OFF_MAX) begin offset <= OFF_MAX; qrs_w <= OFF_MAX - onset; es <= E_P0; end
                        else jj <= jj + 9'd1;
                    end
            E_P0:  begin pw0 <= onset - P_A; pw1 <= onset - P_B; ja <= onset - P_A; es <= E_P1; end
            E_P1:  begin e0 <= m5v; ja <= pw1; es <= E_P2; end
            E_P2:  begin e1 <= m5v; ja <= pw0; best <= 18'd0; jp <= pw0; es <= E_P; end
            E_P:   begin
                        // dev = |2*m5 - e0 - e1|
                        dev_t = $signed({9'd0, m5v, 1'b0}) - $signed({9'd0, e0}) - $signed({9'd0, e1});
                        if (dev_t < 0) dev_t = -dev_t;
                        best_n = best; jp_n = jp;
                        if (dev_t > $signed({6'd0, best})) begin best_n = dev_t[17:0]; jp_n = ja; end
                        best <= best_n; jp <= jp_n;
                        if (ja == pw1) begin p_amp <= best_n[16:3]; pr <= onset - jp_n; es <= E_T0; end
                        else ja <= ja + 9'd1;
                    end
            E_T0:  begin
                        b5 <= base5; ja <= offset + T_A; jt <= offset + T_A; best <= 18'd0; m5b <= base5;
                        tw1 <= ((offset + T_A + T_LEN) > T_MAX) ? T_MAX : (offset + T_A + T_LEN);
                        es <= E_T;
                    end
            E_T:   begin
                        dev_t = $signed({9'd0, m5v}) - $signed({9'd0, b5});
                        if (dev_t < 0) dev_t = -dev_t;
                        best_n = best; jt_n = jt; m5b_n = m5b;
                        if (dev_t > $signed({6'd0, best})) begin best_n = dev_t[17:0]; jt_n = ja; m5b_n = m5v; end
                        best <= best_n; jt <= jt_n; m5b <= m5b_n;
                        if (ja == tw1) begin
                            t_amp <= best_n[16:3]; tpol <= (m5b_n > b5); rt <= jt_n - 9'd140;
                            dv_i <= 4'd0; es <= E_DIV;
                        end else ja <= ja + 9'd1;
                    end
            E_DIV:  begin dv_start <= 1'b1; es <= E_DIVW; end
            E_DIVW: if (dv_done) begin
                        case (dv_i)
                            4'd0: fv[0] <= dv_fv; 4'd1: fv[1] <= dv_fv; 4'd2: fv[2] <= dv_fv;
                            4'd3: fv[3] <= dv_fv; 4'd4: fv[4] <= dv_fv; 4'd5: fv[5] <= dv_fv;
                            default: fv[7] <= dv_fv;
                        endcase
                        if (dv_i == 4'd6) begin
                            fv[6] <= {10'd0, (tpol ^ (pol_cnt >= 0))};
                            fv[8] <= (qrs_w > 9'd1023) ? 11'd1023 : {2'd0, qrs_w};
                            fv[9] <= {5'd0, pr};
                            tnode <= 8'd0; tstep <= 4'd0; es <= E_TREE;
                        end else begin dv_i <= dv_i + 4'd1; es <= E_DIV; end
                    end
            E_TREE: begin
                        if (t_leaf) begin
                            lc_n = t_cls; lp_n = t_pur;
                            leaf_cls <= lc_n; leaf_pur <= lp_n;
                            if (lp_n >= TAU) begin cls_r <= lc_n; early_r <= 1'b1; es <= E_UPD; end
                            else begin early_r <= 1'b0; idx <= 9'd0; mx <= 13'd0; es <= E_NRM; end
                        end else if (tstep == TREE_DEPTH) begin
                            leaf_cls <= 2'd0; leaf_pur <= 8'd0;
                            if (8'd0 >= TAU) begin cls_r <= 2'd0; early_r <= 1'b1; es <= E_UPD; end
                            else begin early_r <= 1'b0; idx <= 9'd0; mx <= 13'd0; es <= E_NRM; end
                        end else begin
                            tnode <= (t_fval <= t_thr) ? t_left : t_right; tstep <= tstep + 4'd1;
                        end
                    end
            E_NRM: begin
                        mxn = ({1'b0, absd(bwm[idx], base)} > mx) ? {1'b0, absd(bwm[idx], base)} : mx;
                        mx <= mxn;
                        if (idx == 9'd359) begin sh <= shamt(mxn); a3 <= 9'd0; idx <= 9'd0; es <= E_DS; end
                        else idx <= idx + 9'd1;
                    end
            E_DS:  begin
                        um[idx[6:0]] <= wsum4[7:0];
                        a3 <= a3 + 9'd3;
                        if (idx == 9'd119) begin oc <= 4'd0; ci <= 7'd0; kk <= 2'd0; bp <= 1'b1; es <= E_C1; end
                        else idx <= idx + 9'd1;
                    end
            E_C1:  if (bp) begin acc <= bias_s; bp <= 1'b0; end
                   else begin
                        acc_n = acc + $signed({{4{prod[19]}}, prod});
                        acc <= acc_n;
                        if (kk == 2'd2) begin
                            y1m[c1_ya] <= clamp127(acc_n >>> CNN_S1);
                            kk <= 2'd0; bp <= 1'b1;
                            if (ci == 7'd119) begin
                                ci <= 7'd0;
                                if (oc == 4'd7) begin oc <= 4'd0; ic <= 3'd0; es <= E_C2; end
                                else oc <= oc + 4'd1;
                            end else ci <= ci + 7'd1;
                        end else kk <= kk + 2'd1;
                    end
            E_C2:  if (bp) begin acc <= bias_s; bp <= 1'b0; end
                   else begin
                        acc_n = acc + $signed({{4{prod[19]}}, prod});
                        acc <= acc_n;
                        if (ic == 3'd3 && kk == 2'd2) begin
                            bp <= 1'b1;
                            // z + residual, clamp at 127
                            outm[c2_ra] <= (clamp127(acc_n >>> CNN_S2) + y1m[c2_ra] > 8'd127) ? 7'd127 : (clamp127(acc_n >>> CNN_S2) + y1m[c2_ra]);
                            kk <= 2'd0; ic <= 3'd0;
                            if (ci == 7'd119) begin
                                ci <= 7'd0;
                                if (oc == 4'd3) begin idx <= 9'd0; bp <= 1'b0; es <= E_FQ; end
                                else oc <= oc + 4'd1;
                            end else ci <= ci + 7'd1;
                        end else begin
                            if (kk == 2'd2) begin kk <= 2'd0; ic <= ic + 3'd1; end else kk <= kk + 2'd1;
                        end
                    end
            E_FQ:  begin
                        vecm[9'd32 + idx] <= fq_v;
                        if (idx == 9'd7) begin pc <= 4'd0; pi <= 7'd0; es <= E_POOL; end
                        else idx <= idx + 9'd1;
                    end
            E_POOL: begin
                        pA_n = (pi == 7'd0) ? pl_o : ((pl_o > pA) ? pl_o : pA);
                        sP_n = ((pi == 7'd0) ? 12'd0 : sP) + ((pi >= 22 && pi <= 37) ? {5'd0, pl_o} : 12'd0);
                        mQ_n = (pi == 7'd0) ? 7'd0 : mQ;
                        if (pi >= 38 && pi <= 53 && pl_o > mQ_n) mQ_n = pl_o;
                        sT_n = ((pi == 7'd0) ? 12'd0 : sT) + ((pi >= 60 && pi <= 91) ? {5'd0, pl_o} : 12'd0);
                        pA <= pA_n; sP <= sP_n; mQ <= mQ_n; sT <= sT_n;
                        if (pi == 7'd119) begin
                            vecm[{1'b0, pc}]           <= {1'b0, pA_n};
                            vecm[6'd8  + {2'b0, pc}]   <= {1'b0, sP_n[10:4]};
                            vecm[6'd16 + {2'b0, pc}]   <= {1'b0, mQ_n};
                            vecm[6'd24 + {2'b0, pc}]   <= {1'b0, sT_n[11:5]};
                            pi <= 7'd0;
                            if (pc == 4'd7) begin fj <= 5'd0; fi <= 6'd0; wfi <= ROM_WF1; bp <= 1'b1; es <= E_FC1; end
                            else pc <= pc + 4'd1;
                        end else pi <= pi + 7'd1;
                    end
            E_FC1: if (bp) begin acc <= bias_s; bp <= 1'b0; end
                   else begin
                        acc_n = acc + $signed({{4{prod[19]}}, prod});
                        acc <= acc_n; wfi <= wfi + 10'd1;
                        if (fi == 6'd39) begin
                            hm[fj[3:0]] <= clamp127(acc_n >>> CNN_S3);
                            fi <= 6'd0; bp <= 1'b1;
                            if (fj == 5'd15) begin fj <= 5'd0; wfi <= ROM_WF2; es <= E_FC2; end
                            else fj <= fj + 5'd1;
                        end else fi <= fi + 6'd1;
                    end
            E_FC2: if (bp) begin acc <= bias_s; bp <= 1'b0; end
                   else begin
                        acc_n = acc + $signed({{4{prod[19]}}, prod});
                        acc <= acc_n; wfi <= wfi + 10'd1;
                        if (fi == 6'd15) begin
                            if (fj[1:0] == 2'd0) lg0 <= acc_n;
                            if (fj[1:0] == 2'd1) lg1 <= acc_n;
                            if (fj[1:0] == 2'd2) lg2 <= acc_n;
                            fi <= 6'd0; bp <= 1'b1;
                            if (fj[1:0] == 2'd2) es <= E_ARG; else fj <= fj + 5'd1;
                        end else fi <= fi + 6'd1;
                    end
            E_ARG: begin
                        cls_r <= (lg2 > lg0 && lg2 > lg1) ? 2'd2 : ((lg1 > lg0) ? 2'd1 : 2'd0);
                        early_r <= 1'b0; es <= E_UPD;
                    end
            E_UPD: begin
                        rr_ref <= refnext(rr_ref, rr_pre_c, n_rr);
                        if (n_rr != 8'd255) n_rr <= n_rr + 8'd1;
                        if (cls_r == 2'd0) begin
                            qrs_ref <= refnext(qrs_ref, {7'd0, qrs_w}, n_mor);
                            p_ref   <= refnext(p_ref,   {2'd0, p_amp}, n_mor);
                            t_ref   <= refnext(t_ref,   {2'd0, t_amp}, n_mor);
                            r_ref   <= refnext(r_ref,   {4'd0, ramp},  n_mor);
                            rt_ref  <= refnext(rt_ref,  {7'd0, rt},    n_mor);
                            if (tpol) begin if (pol_cnt != 5'sd7) pol_cnt <= pol_cnt + 5'sd1; end
                            else begin if (pol_cnt != -5'sd8) pol_cnt <= pol_cnt - 5'sd1; end
                            if (n_mor != 8'd255) n_mor <= n_mor + 8'd1;
                        end
                        es <= E_OUT;
                    end
            E_OUT: begin
                        out_valid <= 1'b1; out_cls <= cls_r; out_early <= early_r; out_drop_sat <= drop_r & dsat_r;
                        es <= E_POP;
                    end
            E_POP: begin
                        c_prev <= c_cur; have_prev <= 1'b1; rp <= rp + 5'd1; es <= E_IDLE;
                    end
            default: es <= E_IDLE;
            endcase
        end
    end
endmodule
