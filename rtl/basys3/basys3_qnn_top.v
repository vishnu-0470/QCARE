// ============================================================================
//  basys3_qnn_top : Digilent Basys 3 (Artix-7 XC7A35T) demo wrapper.
//  - sw[1:0]  selects one of 4 stored beats (MIT-BIH 100, DS2, from Stage 1)
//  - btnC     loads the selected beat into the engine and starts classification
//  - 7-seg    digit 3 shows the AAMI class letter, digit 0 shows the beat index
//  - led[4:0] one-hot class (led[0]=N, led[1]=S, led[2]=V, led[3]=F, led[4]=Q)
//  Board behaviour is NOT verified here: run it on the board and check the
//  displayed class against sim/tb_board.v's expected output first.
// ============================================================================
`timescale 1ns/1ps
module basys3_qnn_top (
    input  wire       clk,         // 100 MHz
    input  wire       btnC,
    input  wire [1:0] sw,
    output wire [4:0] led,
    output reg  [6:0] seg,         // active low, {g,f,e,d,c,b,a}
    output reg        dp,
    output reg  [3:0] an           // active low
);
    localparam N_IN = 3600;
    localparam BEATS = 4;

    // ---- power-on reset ----
    reg [7:0] por = 8'd0;
    wire rst_n = por[7];
    always @(posedge clk) if (!por[7]) por <= por + 8'd1;

    // ---- button: 2-flop sync + ~10 ms debounce + edge ----
    reg b1 = 1'b0, b2 = 1'b0, bstable = 1'b0; reg [19:0] bcnt = 20'd0; reg bprev = 1'b0;
    always @(posedge clk) begin
        b1 <= btnC; b2 <= b1;
        if (b2 != bstable) begin
            bcnt <= bcnt + 20'd1;
            if (bcnt == 20'd999_999) begin bstable <= b2; bcnt <= 20'd0; end
        end else bcnt <= 20'd0;
        bprev <= bstable;
    end
    wire press = bstable & ~bprev;

    // ---- beat memory (4 real beats) ----
    reg [7:0] beats [0:BEATS*N_IN-1];
    initial $readmemh("gen/beats.mem", beats);

    // ---- engine ----
    reg         start, in_we;
    reg  [15:0] in_addr;
    reg  [7:0]  in_data;
    wire        busy, done;
    wire [2:0]  cls;
    wire [119:0] logit_flat;
    qnn_engine #(.PROG_FILE("gen/qnn_prog.mem"), .WROM_FILE("gen/qnn_w.mem"),
                 .BROM_FILE("gen/qnn_b.mem")) u_eng (
        .clk(clk), .rst_n(rst_n), .start(start), .in_we(in_we), .in_addr(in_addr),
        .in_data(in_data), .busy(busy), .done(done), .cls(cls), .logit_flat(logit_flat));

    // ---- control FSM ----
    localparam C_IDLE=0, C_LOAD=1, C_GO=2, C_RUN=3, C_SHOW=4;
    reg [2:0]  cs;
    reg [1:0]  beat;
    reg [11:0] li;
    reg [2:0]  cls_r;
    reg        valid;
    always @(posedge clk) begin
        start <= 1'b0; in_we <= 1'b0;
        if (!rst_n) begin
            cs <= C_IDLE; valid <= 1'b0; cls_r <= 3'd0; li <= 12'd0; beat <= 2'd0;
        end else case (cs)
            C_IDLE: if (press) begin
                        beat <= sw[1:0] < BEATS ? sw[1:0] : 2'd0;
                        li <= 12'd0; valid <= 1'b0; cs <= C_LOAD;
                    end
            C_LOAD: begin
                        in_we <= 1'b1; in_addr <= {4'd0, li};
                        in_data <= beats[beat*N_IN + li];
                        li <= li + 12'd1;
                        if (li == N_IN-1) cs <= C_GO;
                    end
            C_GO:   begin start <= 1'b1; cs <= C_RUN; end
            C_RUN:  if (done) begin cls_r <= cls; valid <= 1'b1; cs <= C_SHOW; end
            C_SHOW: if (press) cs <= C_IDLE;
            default: cs <= C_IDLE;
        endcase
    end
    assign led = valid ? (5'd1 << cls_r) : 5'd0;

    // ---- seven-segment display ----
    function [6:0] letter;          // {g,f,e,d,c,b,a}, 1 = segment ON (before inversion)
        input [2:0] c;
        case (c)
            3'd0: letter = 7'b1010100;   // n
            3'd1: letter = 7'b1101101;   // S
            3'd2: letter = 7'b0111110;   // U  (V is shown as U: 7-seg convention)
            3'd3: letter = 7'b1110001;   // F
            3'd4: letter = 7'b1100111;   // q  (Q)
            default: letter = 7'b0000000;
        endcase
    endfunction
    function [6:0] digit;
        input [3:0] d;
        case (d)
            4'd0: digit = 7'b0111111; 4'd1: digit = 7'b0000110; 4'd2: digit = 7'b1011011;
            4'd3: digit = 7'b1001111; default: digit = 7'b0000000;
        endcase
    endfunction

    reg [16:0] scan;
    always @(posedge clk) if (!rst_n) scan <= 17'd0; else scan <= scan + 17'd1;
    wire [1:0] dsel = scan[16:15];
    always @* begin
        an = 4'b1111; dp = 1'b1; seg = 7'h7F;
        case (dsel)
            2'd0: begin an = 4'b1110; seg = ~digit(beat); end                    // beat index
            2'd3: begin an = 4'b0111; seg = valid ? ~letter(cls_r) : 7'h7F; end  // class letter
            default: begin an = 4'b1111; seg = 7'h7F; end
        endcase
    end
endmodule
