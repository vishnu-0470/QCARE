`timescale 1ns/1ps
// Board-path check: press btnC on sw=0 (beat 0, expected class F -> led[3]), wait for led.
module tb_board;
    reg clk = 0; reg btnC = 0; reg [1:0] sw = 2'd0;
    wire [4:0] led; wire [6:0] seg; wire dp; wire [3:0] an;
    basys3_qnn_top dut (.clk(clk), .btnC(btnC), .sw(sw), .led(led), .seg(seg), .dp(dp), .an(an));
    always #5 clk = ~clk;
    integer t;
    initial begin
        #100;
        @(negedge clk) btnC = 1'b1;
        repeat (1_100_000) @(posedge clk);        // > 10 ms debounce
        @(negedge clk) btnC = 1'b0;
        t = 0;
        while (led == 5'd0 && t < 2_000_000) begin @(posedge clk); t = t + 1; end
        $display("led after run = %b (expect 01000 = class F for beat 0), waited %0d cycles", led, t);
        if (led == 5'b01000) $display("BOARD-PATH TEST: PASS");
        else                 $display("BOARD-PATH TEST: FAIL");
        $finish;
    end
endmodule
