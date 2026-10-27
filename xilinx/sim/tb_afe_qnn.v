`timescale 1ns/1ps
// Compile with xilinx/sim/tb_qnn_checked.v from the project root after Python adapter.
module tb_afe_qnn;
 tb_qnn_checked #(.NV(1),
  .PROG_FILE("xilinx/rtl/qnn/gen/qnn_prog.mem"),
  .WROM_FILE("xilinx/rtl/qnn/gen/qnn_w.mem"),
  .BROM_FILE("xilinx/rtl/qnn/gen/qnn_b.mem"),
  .INPUT_FILE("experiments/afe_adc_checked/vec/qnn_vec_in.mem"),
  .EXPECTED_FILE("experiments/afe_adc_checked/vec/qnn_vec_exp.txt")) checked();
endmodule
