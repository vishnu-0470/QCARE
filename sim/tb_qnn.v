`timescale 1ns/1ps
// Bit-exact check of qnn_engine against the MATLAB integer golden model.
module tb_qnn;
    localparam NV = 8, IN_N = 3600;
    reg clk = 0, rst_n = 0, start = 0, in_we = 0;
    reg [15:0] in_addr = 0; reg [7:0] in_data = 0;
    wire busy, done; wire [2:0] cls; wire [119:0] logit_flat;
    qnn_engine #(.PROG_FILE("gen/qnn_prog.mem"), .WROM_FILE("gen/qnn_w.mem"),
                 .BROM_FILE("gen/qnn_b.mem")) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .in_we(in_we), .in_addr(in_addr),
        .in_data(in_data), .busy(busy), .done(done), .cls(cls), .logit_flat(logit_flat));
    always #5 clk = ~clk;
    reg [7:0] vin [0:NV*IN_N-1];
    integer ecls [0:NV-1]; integer elg [0:NV*5-1];
    integer v, i, fails, cyc, t0, tot_cyc, ok;
    integer fd, rc, c0, l0, l1, l2, l3, l4;
    reg signed [23:0] got;
    initial begin
        $readmemh("vec/qnn_vec_in.mem", vin);
        fd = $fopen("vec/qnn_vec_exp.txt", "r");
        for (v = 0; v < NV; v = v + 1) begin
            rc = $fscanf(fd, "%d %d %d %d %d %d\n", c0, l0, l1, l2, l3, l4);
            ecls[v] = c0; elg[v*5+0] = l0; elg[v*5+1] = l1; elg[v*5+2] = l2;
            elg[v*5+3] = l3; elg[v*5+4] = l4;
        end
        $fclose(fd);
        fails = 0; tot_cyc = 0;
        repeat (4) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
        for (v = 0; v < NV; v = v + 1) begin
            @(negedge clk); in_we = 1;
            for (i = 0; i < IN_N; i = i + 1) begin
                in_addr = i; in_data = vin[v*IN_N + i];
                @(negedge clk);
            end
            in_we = 0;
            @(negedge clk); start = 1; @(negedge clk); start = 0;
            t0 = $time;
            wait (done == 1'b1);
            tot_cyc = tot_cyc + ($time - t0)/10;
            ok = (cls == ecls[v]);
            for (i = 0; i < 5; i = i + 1) begin
                got = logit_flat[24*i +: 24];
                if (got != elg[v*5+i]) ok = 0;
            end
            if (!ok) begin
                fails = fails + 1;
                $display("MISMATCH vec %0d: cls %0d (exp %0d)  logits %0d %0d %0d %0d %0d",
                         v, cls, ecls[v], $signed(logit_flat[23:0]), $signed(logit_flat[47:24]),
                         $signed(logit_flat[71:48]), $signed(logit_flat[95:72]), $signed(logit_flat[119:96]));
            end else
                $display("MATCH    vec %0d: cls %0d  logits %0d %0d %0d %0d %0d", v, cls,
                         elg[v*5], elg[v*5+1], elg[v*5+2], elg[v*5+3], elg[v*5+4]);
            @(negedge clk);
        end
        $display("RESULT: %0d / %0d vectors bit-exact, mean %0d cycles per beat (engine only)",
                 NV - fails, NV, tot_cyc / NV);
        $finish;
    end
endmodule
