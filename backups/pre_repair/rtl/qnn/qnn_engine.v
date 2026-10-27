// ============================================================================
//  qnn_engine : integer executor for the 4W/8A Inception-ResNeXt classifier.
//  Bit-exact with matlab/qat/qnn_int_forward.m:
//    accumulator = bias + sum(x * w)            (32-bit signed)
//    requant     = (acc + 2^(s-1)) >>> s        if s > 0   (round half up)
//                  acc <<< (-s)                 if s <= 0
//    activation  = clamp(requant, 0, 127)       for conv / add / gap outputs
//  Synthesis-friendly: every RAM / ROM read is REGISTERED (one-cycle latency),
//  so Yosys/Vivado infer block RAM, and the activation RAM has a single write
//  port.  Reads are pipelined: an issue state starts a read, the value is used
//  one cycle later in the accumulate/reduce stage.
//  The CWT map is an INPUT (computed by MATLAB Stage 1); the CWT itself is not
//  in hardware yet.
// ============================================================================
`timescale 1ns/1ps
module qnn_engine #(
    parameter NOPS      = 22,
    parameter NW        = 9144,
    parameter NB        = 149,
    parameter PROG_FILE = "gen/qnn_prog.mem",
    parameter WROM_FILE = "gen/qnn_w.mem",
    parameter BROM_FILE = "gen/qnn_b.mem"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        in_we,        // write one input byte (only while idle)
    input  wire [15:0] in_addr,
    input  wire [7:0]  in_data,
    output reg         busy,
    output reg         done,         // one-cycle pulse when logits are valid
    output reg  [2:0]  cls,          // 0 N, 1 S, 2 V, 3 F, 4 Q
    output reg  [119:0] logit_flat   // logit[j] = logit_flat[24*j +: 24], signed
);
    localparam NF = 18;
    localparam F_OP=0, F_SRC=1, F_SRC2=2, F_DST=3, F_CI=4, F_CO=5, F_H=6, F_W=7,
               F_HO=8, F_WO=9, F_K=10, F_S=11, F_P=12, F_WB=13, F_BB=14, F_SH=15,
               F_SA=16, F_HALF=17;
    localparam OP_END=0, OP_CONV=1, OP_POOL=2, OP_ADD=3, OP_GAP=4, OP_FC=5;

    // ---------------- memories ----------------
    reg [15:0]        prog [0:NOPS*NF-1];   // small: asynchronous read is fine
    reg signed [15:0] wrom [0:NW-1];
    reg signed [15:0] brom [0:NB-1];
    reg signed [7:0]  ram  [0:65535];
    initial begin
        $readmemh(PROG_FILE, prog);
        $readmemh(WROM_FILE, wrom);
        $readmemh(BROM_FILE, brom);
    end

    // single activation-RAM write port
    reg        wr_en;
    reg [15:0] wr_addr;
    reg signed [7:0] wr_data;
    always @(posedge clk) if (wr_en) ram[wr_addr] <= wr_data;

    // registered read outputs
    reg signed [7:0]  ram_q;   // activation read  (1-cycle latency)
    reg signed [15:0] wq;      // weight read      (1-cycle latency)
    reg signed [15:0] bq;      // bias read        (1-cycle latency)

    // ---------------- latched op fields ----------------
    reg [15:0] op_r, src_r, src2_r, dst_r, ci_r, co_r, h_r, w_r, ho_r, wo_r;
    reg [15:0] k_r, s_r, p_r, wb_r, bb_r, srca_r, half_r;
    reg signed [15:0] sh_r;
    reg [15:0] pc;

    // ---------------- control ----------------
    localparam S_IDLE=0, S_FETCH=1, S_DISPATCH=2,
               S_CV_B1=3, S_CV_B2=4, S_CV_MAC=5, S_CV_FL=6, S_CV_WB=7,
               S_PL_I=8, S_PL_FL=9, S_PL_WR=10,
               S_AD_A=11, S_AD_B=12, S_AD_W=13,
               S_GP_I=14, S_GP_FL=15, S_GP_WR=16,
               S_FC_B1=17, S_FC_B2=18, S_FC_I=19, S_FC_FL=20, S_FC_WR=21,
               S_NEXT=22, S_DONE=23;
    reg [4:0] st;

    integer co_i, oh, ow, ci_i, kh, kw, q, pix, ch, ci_f, hw;
    reg signed [31:0] acc, mx, sum, aval;
    reg vld_mac;         // accumulate stage active (conv / fc)
    reg vld_red;         // reduction stage active (pool / gap)
    reg vfirst;          // first element of a reduction
    reg tmp_in;          // convolution tap is inside the padded input
    integer ih, iw, ra, wa, idx, jx;
    // one read site per memory: states choose the address, the end of the block performs the read
    integer rda, rwa, rba; reg rde, rwe, rbe;
    reg signed [23:0] tbest, lv;
    reg [2:0] cbest;

    function [7:0] clamp_act;
        input signed [31:0] v;
        begin
            if (v < 0) clamp_act = 8'd0;
            else if (v > 127) clamp_act = 8'd127;
            else clamp_act = v[7:0];
        end
    endfunction

    function signed [31:0] requant;
        input signed [31:0] a;
        input signed [15:0] s;
        reg signed [31:0] r;
        begin
            if (s > 0) r = (a + (32'sd1 <<< (s - 1))) >>> s;
            else       r = a <<< (-s);
            requant = r;
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; cls <= 3'd0; pc <= 16'd0;
            logit_flat <= 120'd0; wr_en <= 1'b0; vld_mac <= 1'b0; vld_red <= 1'b0;
        end else begin
            // ---- defaults ----
            done <= 1'b0; wr_en <= 1'b0; vld_mac <= 1'b0; vld_red <= 1'b0;
            rde = 1'b0; rwe = 1'b0; rbe = 1'b0;
            // ---- reduction / accumulate stages (use values registered last cycle) ----
            if (vld_mac) acc <= acc + ram_q * wq;
            if (vld_red) begin
                if (op_r == OP_POOL) mx  <= vfirst ? ram_q : ((ram_q > mx) ? ram_q : mx);
                if (op_r == OP_GAP)  sum <= vfirst ? ram_q : sum + ram_q;
            end
            // ---- input load (idle only) ----
            if (in_we && st == S_IDLE) begin wr_en <= 1'b1; wr_addr <= in_addr; wr_data <= in_data; end

            case (st)
            S_IDLE: if (start) begin pc <= 16'd0; busy <= 1'b1; st <= S_FETCH; end
            S_FETCH: begin
                op_r <= prog[pc*NF + F_OP];   src_r <= prog[pc*NF + F_SRC];
                src2_r <= prog[pc*NF + F_SRC2]; dst_r <= prog[pc*NF + F_DST];
                ci_r <= prog[pc*NF + F_CI];   co_r <= prog[pc*NF + F_CO];
                h_r <= prog[pc*NF + F_H];     w_r <= prog[pc*NF + F_W];
                ho_r <= prog[pc*NF + F_HO];   wo_r <= prog[pc*NF + F_WO];
                k_r <= prog[pc*NF + F_K];     s_r <= prog[pc*NF + F_S];
                p_r <= prog[pc*NF + F_P];     wb_r <= prog[pc*NF + F_WB];
                bb_r <= prog[pc*NF + F_BB];   sh_r <= prog[pc*NF + F_SH];
                srca_r <= prog[pc*NF + F_SA]; half_r <= prog[pc*NF + F_HALF];
                st <= S_DISPATCH;
            end
            S_DISPATCH: begin
                co_i = 0; oh = 0; ow = 0; pix = 0; q = 0; ch = 0;
                case (op_r)
                    OP_END:  st <= S_DONE;
                    OP_CONV: st <= S_CV_B1;
                    OP_POOL: st <= S_PL_I;
                    OP_ADD:  st <= S_AD_A;
                    OP_GAP:  st <= S_GP_I;
                    OP_FC:   st <= S_FC_B1;
                    default: st <= S_DONE;
                endcase
            end
            // ---------------------------------------------- CONV
            S_CV_B1: begin                       // issue bias read
                rbe = 1'b1; rba = bb_r + co_i;
                ci_i = 0; kh = 0; kw = 0;
                st <= S_CV_B2;
            end
            S_CV_B2: begin                       // bias available: start accumulator
                acc <= bq;
                st <= S_CV_MAC;
            end
            S_CV_MAC: begin                      // issue one tap
                ih = oh*s_r + kh - p_r;
                iw = ow*s_r + kw - p_r;
                tmp_in = (ih >= 0) && (ih < h_r) && (iw >= 0) && (iw < w_r);
                ra = tmp_in ? (src_r + ci_i*h_r*w_r + ih*w_r + iw) : 0;
                wa = wb_r + ((co_i*ci_r + ci_i)*k_r + kh)*k_r + kw;
                rde = tmp_in; rda = ra;
                rwe = 1'b1; rwa = wa;
                vld_mac <= tmp_in;
                if (kw == k_r-1) begin
                    kw = 0;
                    if (kh == k_r-1) begin
                        kh = 0;
                        if (ci_i == ci_r-1) begin ci_i = 0; st <= S_CV_FL; end
                        else ci_i = ci_i + 1;
                    end else kh = kh + 1;
                end else kw = kw + 1;
            end
            S_CV_FL: st <= S_CV_WB;              // absorb the last tap
            S_CV_WB: begin
                wr_en   <= 1'b1;
                wr_addr <= dst_r + co_i*ho_r*wo_r + oh*wo_r + ow;
                wr_data <= $signed(clamp_act(requant(acc, sh_r)));
                if (ow == wo_r-1) begin
                    ow = 0;
                    if (oh == ho_r-1) begin
                        oh = 0;
                        if (co_i == co_r-1) st <= S_NEXT;
                        else begin co_i = co_i + 1; st <= S_CV_B1; end
                    end else begin oh = oh + 1; st <= S_CV_B1; end
                end else begin ow = ow + 1; st <= S_CV_B1; end
            end
            // ---------------------------------------------- POOL 2x2 max
            S_PL_I: begin                        // issue one of 4 reads
                idx = src_r + co_i*h_r*w_r + (2*oh + q/2)*w_r + (2*ow + q%2);
                rde = 1'b1; rda = idx;
                vld_red <= 1'b1; vfirst <= (q == 0);
                q = q + 1;
                if (q == 4) st <= S_PL_FL;
            end
            S_PL_FL: begin                       // absorb the last read
                st <= S_PL_WR;
            end
            S_PL_WR: begin
                wr_en   <= 1'b1;
                wr_addr <= dst_r + co_i*ho_r*wo_r + oh*wo_r + ow;
                wr_data <= mx[7:0];
                q = 0;
                if (ow == wo_r-1) begin
                    ow = 0;
                    if (oh == ho_r-1) begin
                        oh = 0;
                        if (co_i == co_r-1) st <= S_NEXT;
                        else begin co_i = co_i + 1; st <= S_PL_I; end
                    end else begin oh = oh + 1; st <= S_PL_I; end
                end else begin ow = ow + 1; st <= S_PL_I; end
            end
            // ---------------------------------------------- ADD + concat
            S_AD_A: begin                        // read residual A
                hw = h_r*w_r;
                rde = 1'b1; rda = srca_r + co_i*hw + pix;
                st <= S_AD_B;
            end
            S_AD_B: begin                        // capture A, read B
                hw = h_r*w_r;
                aval <= ram_q;
                rde = 1'b1;
                if (co_i < half_r) rda = src_r  + co_i*hw + pix;
                else               rda = src2_r + co_i*hw + pix;
                st <= S_AD_W;
            end
            S_AD_W: begin                        // write clamp(A + B)
                hw = h_r*w_r;
                wr_en   <= 1'b1;
                wr_addr <= dst_r + co_i*hw + pix;
                wr_data <= $signed(clamp_act(aval + ram_q));
                st <= S_AD_A;
                pix = pix + 1;
                if (pix == hw) begin
                    pix = 0; co_i = co_i + 1;
                    if (co_i == co_r) st <= S_NEXT;
                end
            end
            // ---------------------------------------------- GAP
            S_GP_I: begin                        // issue one element
                hw = h_r*w_r;
                rde = 1'b1; rda = src_r + co_i*hw + pix;
                vld_red <= 1'b1; vfirst <= (pix == 0);
                pix = pix + 1;
                if (pix == hw) st <= S_GP_FL;
            end
            S_GP_FL: st <= S_GP_WR;
            S_GP_WR: begin
                hw = h_r*w_r;
                wr_en   <= 1'b1;
                wr_addr <= dst_r + co_i;
                // floor(mean + 0.5) with HW a power of two: (2*sum + HW) >> (log2(HW)+1)
                wr_data <= $signed(clamp_act((2*sum + hw) >>> k_r));   // k_r = log2(HW)+1 (exporter)
                pix = 0; co_i = co_i + 1;
                if (co_i == co_r) st <= S_NEXT; else st <= S_GP_I;
            end
            // ---------------------------------------------- FC
            S_FC_B1: begin
                rbe = 1'b1; rba = bb_r + co_i;
                ci_i = 0;
                st <= S_FC_B2;
            end
            S_FC_B2: begin
                acc <= bq;
                st <= S_FC_I;
            end
            S_FC_I: begin                        // issue one tap
                rde = 1'b1; rda = src_r + ci_i;
                rwe = 1'b1; rwa = wb_r + co_i*ci_r + ci_i;
                vld_mac <= 1'b1;
                ci_i = ci_i + 1;
                if (ci_i == ci_r) st <= S_FC_FL;
            end
            S_FC_FL: st <= S_FC_WR;
            S_FC_WR: begin
                logit_flat[24*co_i +: 24] <= acc[23:0];
                co_i = co_i + 1;
                if (co_i == co_r) st <= S_NEXT; else st <= S_FC_B1;
            end
            // ---------------------------------------------- sequencing
            S_NEXT: begin
                pc <= pc + 16'd1; st <= S_FETCH;
            end
            S_DONE: begin
                // first-maximum argmax over the five logits (strict >, as in MATLAB)
                tbest = $signed(logit_flat[23:0]);
                cbest = 3'd0;
                for (jx = 1; jx < 5; jx = jx + 1) begin
                    lv = $signed(logit_flat[24*jx +: 24]);
                    if (lv > tbest) begin tbest = lv; cbest = jx; end
                end
                cls <= cbest;
                busy <= 1'b0; done <= 1'b1; st <= S_IDLE;
            end
            default: st <= S_IDLE;
            endcase
            // ---- the only read ports: one per memory ----
            if (rde) ram_q <= ram[rda];
            if (rwe) wq    <= wrom[rwa];
            if (rbe) bq    <= brom[rba];
        end
    end
endmodule
