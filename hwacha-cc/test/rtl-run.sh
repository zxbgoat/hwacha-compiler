#!/bin/bash
# usage: rtl-run.sh <outfile> <binary>... : run binaries one after another on the RTL simulator (absolute or cwd-relative paths)
SIM=$HOME/hwacha-compiler/chipyard/sims/verilator/simulator-chipyard.harness-HwachaRocketConfig
out=$1; shift; : > "$out"
for b in "$@"; do
  echo "==> $b start $(date +%T)" >> "$out"
  $SIM +permissive +max-cycles=4000000000 +loadmem="$b" +permissive-off "$b" 2>&1 | grep --line-buffered -v "^\[UART\]" >> "$out"
  echo "==> $b end $(date +%T) rc=${PIPESTATUS[0]}" >> "$out"
done
