`timescale 1ns/1ps
module tb_accel;
    parameter NS     = 21600;
    parameter PERIOD = 200;                      // clock cycles per ADC sample
    parameter STREAM = "xilinx/sim/rec_stream.mem";
    parameter LOG    = "xilinx/sim/rec_rtl.txt";
    parameter CYCLOG = "xilinx/sim/rec_cycles.txt";
    parameter WROM   = "xilinx/rtl/gen/wrom.mem";
    parameter TREE   = "xilinx/rtl/gen/tree.mem";
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg s_valid = 0; reg [11:0] s_data = 0;
    wire out_valid; wire [1:0] out_cls; wire out_early, out_drop_sat, busy, err_fifo, err_ring;
    reg [11:0] stim [0:NS-1];
    ecg_afe_accel_top #(.WROM_FILE(WROM), .TREE_FILE(TREE)) dut (
        .clk(clk), .rst_n(rst_n), .s_valid(s_valid), .s_data(s_data),
        .out_valid(out_valid), .out_cls(out_cls), .out_early(out_early),
        .out_drop_sat(out_drop_sat), .busy(busy), .err_fifo(err_fifo), .err_ring(err_ring));
    integer f, fc, i, cyc, t0, nbeats;
    initial begin
        $readmemh(STREAM, stim);
        f = $fopen(LOG, "w"); fc = $fopen(CYCLOG, "w"); nbeats = 0; cyc = 0;
        repeat (100) @(posedge clk); rst_n <= 1; repeat (10) @(posedge clk);
        for (i = 0; i < NS; i = i + 1) begin
            @(posedge clk); s_data <= stim[i]; s_valid <= 1;
            @(posedge clk); s_valid <= 0;
            repeat (PERIOD - 2) @(posedge clk);
        end
        repeat (60000) @(posedge clk);
        $fclose(f); $fclose(fc);
        $display("TB done: %0d beats logged, err_fifo=%0b err_ring=%0b", nbeats, err_fifo, err_ring);
        $finish;
    end
    always @(posedge clk) cyc <= cyc + 1;
    reg busy_d = 0;
    always @(posedge clk) begin
        busy_d <= busy;
        if (busy && !busy_d) t0 <= cyc;
        if (out_valid) begin
            nbeats = nbeats + 1;
            $fwrite(fc, "%0d %0d %0d\n", cyc - t0, out_cls, out_early);
            if (out_cls == 2'd3)
                $fwrite(f, "D %0d %0d %0d\n", dut.c_cur, dut.R, dut.base);
            else begin
                $fwrite(f, "B %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                    dut.c_cur, dut.R, out_cls, out_early, dut.base, dut.qrs_w, dut.p_amp, dut.t_amp,
                    dut.ramp, dut.tpol, dut.rt, dut.pr, out_early ? 3'd0 : dut.sh);
                $fwrite(f, " %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                    dut.fv[0], dut.fv[1], dut.fv[2], dut.fv[3], dut.fv[4], dut.fv[5], dut.fv[6], dut.fv[7], dut.fv[8], dut.fv[9]);
                if (out_early) $fwrite(f, " %0d %0d 0 0 0\n", dut.leaf_cls, dut.leaf_pur);
                else $fwrite(f, " %0d %0d %0d %0d %0d\n", dut.leaf_cls, dut.leaf_pur, dut.lg0, dut.lg1, dut.lg2);
            end
        end
    end
endmodule
