# User-run flow; NOT TESTED here because Vivado is unavailable.
# Run: vivado -mode batch -source fpga/vivado_qcare.tcl
# Uses the frozen trained stored-map demo, and never programs a board.
set root [file normalize [file join [file dirname [info script]] ..]]
set maps [file join $root experiments ds1_current board_export]
set out [file join $root experiments vivado_trained]
file mkdir $out
cd $maps
create_project -in_memory -part xc7a35tcpg236-1
read_verilog [file join $root rtl qnn qnn_engine.v]
read_verilog [file join $root rtl basys3 basys3_qnn_top.v]
synth_design -top basys3_qnn_top -part xc7a35tcpg236-1
read_xdc [file join $root rtl basys3 basys3.xdc]
report_utilization -file [file join $out utilization_synth.rpt]
write_checkpoint -force [file join $out synthesized.dcp]
opt_design
place_design
route_design
report_utilization -file [file join $out utilization_route.rpt]
report_timing_summary -report_unconstrained -file [file join $out timing.rpt]
report_drc -file [file join $out drc.rpt]
# Vectorless power is an ESTIMATE; provide real switching activity for rigor.
report_power -file [file join $out power_vectorless_estimated.rpt]
write_checkpoint -force [file join $out routed.dcp]
set paths [get_timing_paths -max_paths 1]
if {[llength $paths] == 0} { error "No constrained timing path was returned" }
if {[get_property SLACK [lindex $paths 0]] < 0} { error "Timing target failed; see timing.rpt" }
write_bitstream -force [file join $out qcare_trained.bit]
