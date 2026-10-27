`timescale 1ns/1ps
// Runs one Cadence-derived CWT map through the existing integer QNN engine.
// Generate vec/afe_qnn_vec_in.mem with sim/prepare_afe_qnn_input.m first.
module tb_afe_qnn;
    localparam IN_N = 3600;
    reg clk = 0, rst_n = 0, start = 0, in_we = 0;
    reg [15:0] in_addr = 0;
    reg [7:0] in_data = 0;
    wire busy, done;
    wire [2:0] cls;
    wire [119:0] logit_flat;
    reg [7:0] vin [0:IN_N-1];
    integer i, timeout_cycles;
    reg signed [23:0] logit_value;

    qnn_engine #(.PROG_FILE("gen/qnn_prog.mem"), .WROM_FILE("gen/qnn_w.mem"),
                 .BROM_FILE("gen/qnn_b.mem")) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .in_we(in_we), .in_addr(in_addr),
        .in_data(in_data), .busy(busy), .done(done), .cls(cls), .logit_flat(logit_flat));

    always #5 clk = ~clk;

    initial begin
        $readmemh("vec/afe_qnn_vec_in.mem", vin);
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        @(negedge clk);
        in_we = 1;
        for (i = 0; i < IN_N; i = i + 1) begin
            in_addr = i;
            in_data = vin[i];
            @(negedge clk);
        end
        in_we = 0;
        start = 1;
        @(negedge clk);
        start = 0;

        timeout_cycles = 0;
        while (done !== 1'b1 && timeout_cycles < 200000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end
        if (done !== 1'b1) begin
            $display("ERROR: QNN did not finish within timeout");
            $finish;
        end
        #1;
        $display("Cadence waveform inference: class %0d (0=N, 1=S, 2=V, 3=F, 4=Q)", cls);
        for (i = 0; i < 5; i = i + 1) begin
            logit_value = logit_flat[24*i +: 24];
            $display("logit[%0d] = %0d", i, logit_value);
        end
        $finish;
    end
endmodule
