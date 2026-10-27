# Run the existing fusion engine and Basys 3 testbenches with Vivado XSim.
# From the Vivado Tcl Console, run:
#   cd C:/QCARE_FPGA
#   source xilinx/fpga/vivado_xsim_fusion90.tcl
#
# Each invocation gets a fresh report directory; this keeps existing evidence
# and checked model exports intact.

set root [file normalize [file join [file dirname [info script]] .. ..]]
set rtl [file join $root xilinx rtl qnn qnn_engine.v]
set fusion_tb [file join $root xilinx sim tb_qnn_fusion_checked.v]
set board_rtl [file join $root xilinx rtl basys3 basys3_qnn_fusion_top.v]
set board_tb [file join $root xilinx sim tb_board_fusion_checked.v]
set fusion_export [file join $root experiments accuracy90_fusion_mlp export]
set board_export [file join $root experiments accuracy90_fusion_mlp board_export]
set run_id [clock format [clock seconds] -format %Y%m%d_%H%M%S]
set output_root [file join $root reports vivado_xsim $run_id]

proc run_stage {cmd log_path} {
    set rc [catch {exec {*}$cmd 2>@1} output options]
    set f [open $log_path w]
    puts $f [join $cmd " "]
    puts $f $output
    close $f
    if {$rc} {
        return -options $options $output
    }
    puts $output
    return $output
}

proc copy_required {source destination} {
    if {![file exists $source]} {
        error "Required simulation input is missing: $source"
    }
    file copy -force $source $destination
}

proc run_fusion {root output_dir rtl testbench export_dir} {
    file mkdir [file join $output_dir gen]
    file mkdir [file join $output_dir vec]
    foreach name {qnn_b.mem qnn_prog.mem qnn_prog.mem.wide qnn_w.mem} {
        copy_required [file join $export_dir gen $name] [file join $output_dir gen $name]
    }
    foreach name {qnn_vec_in.mem qnn_vec_exp.txt} {
        copy_required [file join $export_dir vec $name] [file join $output_dir vec $name]
    }

    cd $output_dir
    run_stage [list xvlog -sv -work xil_defaultlib $rtl $testbench] [file join $output_dir compile.log]
    run_stage [list xelab -debug typical -s fusion_sim xil_defaultlib.tb_qnn_fusion_checked] [file join $output_dir elaborate.log]
    set output [run_stage [list xsim fusion_sim -runall -log xsim.log] [file join $output_dir run.log]]
    if {[string first "PASS 10/10 vectors" $output] < 0} {
        error "Fusion XSim run finished without its PASS marker. See [file join $output_dir run.log]"
    }
    puts "Fusion test passed. Logs: $output_dir"
}

proc run_board {root output_dir rtl board_rtl testbench export_dir} {
    file mkdir [file join $output_dir gen]
    foreach name {beats.mem qnn_b.mem qnn_prog.mem qnn_prog.mem.wide qnn_w.mem} {
        copy_required [file join $export_dir gen $name] [file join $output_dir gen $name]
    }
    copy_required [file join $export_dir expected.txt] [file join $output_dir expected.txt]

    cd $output_dir
    run_stage [list xvlog -sv -work xil_defaultlib $rtl $board_rtl $testbench] [file join $output_dir compile.log]
    run_stage [list xelab -debug typical -s board_sim xil_defaultlib.tb_board_fusion_checked] [file join $output_dir elaborate.log]
    set output [run_stage [list xsim board_sim -runall -log xsim.log] [file join $output_dir run.log]]
    if {[string first "PASS 4/4 board switch selections" $output] < 0} {
        error "Board XSim run finished without its PASS marker. See [file join $output_dir run.log]"
    }
    puts "Board test passed. Logs: $output_dir"
}

file mkdir $output_root
set original_dir [pwd]
set result [catch {
    run_fusion $root [file join $output_root fusion] $rtl $fusion_tb $fusion_export
    cd $original_dir
    run_board $root [file join $output_root board] $rtl $board_rtl $board_tb $board_export
} message options]
cd $original_dir
if {$result} {
    return -options $options $message
}
puts "Both XSim checks passed. Results are under: $output_root"
