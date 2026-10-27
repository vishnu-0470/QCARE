`timescale 1ns/1ps
// Compile with sim/tb_qnn_checked.v. Run from rtl/qnn after Python adapter.
module tb_afe_qnn;
 tb_qnn_checked #(.NV(1),
  .INPUT_FILE("../../experiments/afe_adc_checked/vec/qnn_vec_in.mem"),
  .EXPECTED_FILE("../../experiments/afe_adc_checked/vec/qnn_vec_exp.txt")) checked();
endmodule
