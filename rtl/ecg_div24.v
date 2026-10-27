// Restoring divider, 24-bit unsigned, 24 cycles. q = floor(num/den), den != 0.
`timescale 1ns/1ps
module ecg_div24 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [23:0] num,
    input  wire [23:0] den,
    output reg         done,
    output reg  [23:0] q
);
    reg [23:0] n;
    reg [24:0] r;
    reg [23:0] d;
    reg [4:0]  cnt;
    reg        run;
    wire [24:0] r_sh = {r[23:0], n[23]};
    wire        ge   = (r_sh >= {1'b0, d});
    always @(posedge clk) begin
        if (!rst_n) begin
            run <= 1'b0; done <= 1'b0; q <= 24'd0; n <= 24'd0; r <= 25'd0; d <= 24'd1; cnt <= 5'd0;
        end else begin
            done <= 1'b0;
            if (start && !run) begin
                run <= 1'b1; n <= num; d <= den; r <= 25'd0; q <= 24'd0; cnt <= 5'd24;
            end else if (run) begin
                r <= ge ? (r_sh - {1'b0, d}) : r_sh;
                q <= {q[22:0], ge};
                n <= {n[22:0], 1'b0};
                cnt <= cnt - 5'd1;
                if (cnt == 5'd1) begin run <= 1'b0; done <= 1'b1; end
            end
        end
    end
endmodule
