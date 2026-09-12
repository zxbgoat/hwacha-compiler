#!/bin/bash
# usage: run-rtl.sh <binary> <outfile>
SIM=$HOME/hwacha-compiler/chipyard/sims/verilator/simulator-chipyard.harness-HwachaRocketConfig
cd "$(dirname "$0")"
echo "==> $1 start $(date +%T)" > "$2"
$SIM +permissive +max-cycles=4000000000 +loadmem="$1" +permissive-off "$1" 2>&1 | grep --line-buffered -v "^\[UART\]" >> "$2"
echo "==> $1 end $(date +%T) rc=${PIPESTATUS[0]}" >> "$2"
