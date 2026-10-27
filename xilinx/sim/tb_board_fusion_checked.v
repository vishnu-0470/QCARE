`timescale 1ns/1ps
// Four simultaneous wrapper instances check each switch-selected stored beat.
module tb_board_fusion_checked;
 reg clk=0,btnC=0; always #5 clk=~clk;
 integer expected[0:23];integer fd,i,rc,c,l0,l1,l2,l3,l4;
 reg [3:0] checked=0;
 function [6:0] letter;
  input integer c;
  case(c) 0:letter=7'b1010100;1:letter=7'b1101101;2:letter=7'b0111110;
   3:letter=7'b1110001;4:letter=7'b1100111;default:letter=0;endcase
 endfunction
 function [6:0] digit;
  input integer d;
  case(d) 0:digit=7'b0111111;1:digit=7'b0000110;2:digit=7'b1011011;3:digit=7'b1001111;default:digit=0;endcase
 endfunction
 initial begin
  fd=$fopen("expected.txt","r");if(!fd)$fatal(1,"Missing board golden file");
  for(i=0;i<4;i=i+1)begin
   rc=$fscanf(fd,"%d %d %d %d %d %d\n",c,l0,l1,l2,l3,l4);
   if(rc!=6)$fatal(1,"Malformed expected board row");
   expected[6*i]=c;expected[6*i+1]=l0;expected[6*i+2]=l1;expected[6*i+3]=l2;expected[6*i+4]=l3;expected[6*i+5]=l4;
  end
  $fclose(fd);repeat(1000)@(negedge clk);btnC=1;
  repeat(1100000)@(negedge clk);btnC=0;
  wait(checked===4'b1111);$display("PASS 4/4 board switch selections: logits, LEDs and display segments");$finish;
 end
 initial begin repeat(3500000)@(posedge clk);$fatal(1,"Board wrapper timeout");end
 genvar b;
 generate for(b=0;b<4;b=b+1)begin: g
  wire [4:0] led;wire [6:0] seg;wire dp;wire [3:0] an;
  localparam [1:0] SWITCH=b;
  basys3_qnn_fusion_top dut(clk,btnC,SWITCH,led,seg,dp,an);
  integer j,n;reg seen_index,seen_class;reg signed [23:0] lg;
  initial begin
   wait(led!==5'd0 && (^led)!==1'bx);@(negedge clk);
   if(led !== (5'd1 << expected[6*b]))$fatal(1,"Wrong LED for beat %0d",b);
   for(j=0;j<5;j=j+1)begin
    lg=dut.logit_flat[j*24+:24];
    if(lg!==expected[6*b+j+1])$fatal(1,"Board logit mismatch beat %0d class %0d",b,j);
   end
   seen_index=0;seen_class=0;
   for(n=0;n<131072;n=n+1)begin
    @(negedge clk);
    if((^an)===1'bx || (^seg)===1'bx)$fatal(1,"Unknown display output beat %0d",b);
    if(dp!==1'b1)$fatal(1,"Decimal point must be off");
    if(an==4'b1110)begin seen_index=1;if(seg!==(~digit(b)))$fatal(1,"Wrong displayed index");end
    if(an==4'b0111)begin seen_class=1;if(seg!==(~letter(expected[6*b])))$fatal(1,"Wrong displayed class");end
   end
   if(!seen_index || !seen_class)$fatal(1,"Display scan did not visit required digits");
   $display("BOARD MATCH switch=%0d class=%0d LED=%b",b,expected[6*b],led);checked[b]=1;
  end
 end endgenerate
endmodule
