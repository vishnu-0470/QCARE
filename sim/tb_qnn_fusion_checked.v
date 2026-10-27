`timescale 1ns/1ps
module tb_qnn_fusion_checked;
 parameter NV=10;
 parameter NOPS=23,NW=10248,NB=181;
 parameter PROG_FILE="gen/qnn_prog.mem",WROM_FILE="gen/qnn_w.mem",BROM_FILE="gen/qnn_b.mem";
 parameter INPUT_FILE="vec/qnn_vec_in.mem",EXPECTED_FILE="vec/qnn_vec_exp.txt";
 reg clk=0,rst_n=0,start=0,in_we=0;reg [15:0] in_addr=0;reg [7:0] in_data=0;
 wire busy,done;wire [2:0] cls;wire [119:0] logit_flat;
 qnn_engine #(.NOPS(NOPS),.NW(NW),.NB(NB),.PROG_FILE(PROG_FILE),.WROM_FILE(WROM_FILE),.BROM_FILE(BROM_FILE)) dut(clk,rst_n,start,in_we,in_addr,in_data,busy,done,cls,logit_flat);
 always #5 clk=~clk;
 reg [7:0] vin[0:NV*3616-1];integer expected[0:NV*6-1];integer v,i,fd,rc,j,cycles,waited,c0,l0,l1,l2,l3,l4;
 reg signed [23:0] got;
 initial begin
  $readmemh(INPUT_FILE,vin);
  for(i=0;i<NV*3616;i=i+1)if((^vin[i])===1'bx)$fatal(1,"Missing/unknown input token %0d",i);
  fd=$fopen(EXPECTED_FILE,"r");if(!fd)$fatal(1,"Cannot open expected logits");
  for(v=0;v<NV;v=v+1)begin
   rc=$fscanf(fd,"%d %d %d %d %d %d\n",c0,l0,l1,l2,l3,l4);
   expected[v*6]=c0;expected[v*6+1]=l0;expected[v*6+2]=l1;expected[v*6+3]=l2;expected[v*6+4]=l3;expected[v*6+5]=l4;
   if(rc!=6)$fatal(1,"Malformed expected row %0d",v);
  end
  $fclose(fd);repeat(4)@(negedge clk);rst_n=1;repeat(2)@(negedge clk);
  for(v=0;v<NV;v=v+1)begin
   in_we=1;
   for(i=0;i<3616;i=i+1)begin in_addr=(i<3600)?i:32064+i-3600;in_data=vin[v*3616+i];@(negedge clk);end
   in_we=0;@(negedge clk);start=1;@(negedge clk);start=0;waited=0;
   while(done!==1'b1 && waited<2000000)begin @(negedge clk);waited=waited+1;end
   if(done!==1'b1)$fatal(1,"QNN timeout vector %0d",v);
   if(cls!==expected[v*6])$fatal(1,"Class mismatch vector %0d",v);
   for(j=0;j<5;j=j+1)begin
    got=logit_flat[24*j+:24];if(got!==expected[v*6+j+1])$fatal(1,"Logit mismatch vector %0d class %0d expected=%0d got=%0d",v,j,expected[v*6+j+1],got);
   end
   $display("MATCH vector=%0d class=%0d cycles=%0d logits=%0d,%0d,%0d,%0d,%0d",v,cls,waited,$signed(logit_flat[23:0]),$signed(logit_flat[47:24]),$signed(logit_flat[71:48]),$signed(logit_flat[95:72]),$signed(logit_flat[119:96]));
   @(negedge clk);
  end
  $display("PASS %0d/%0d vectors: all signed int24 logits and classes bit-exact",NV,NV);$finish;
 end
endmodule
