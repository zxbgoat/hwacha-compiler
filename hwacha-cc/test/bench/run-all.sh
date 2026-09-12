#!/bin/bash
# Run benchmark variants on the RTL simulator in parallel; results in results/*.log
# usage: ./run-all.sh [N]   (default 1024)
N=${1:-1024}
cd "$(dirname "$0")"; mkdir -p results
run() { ./run-rtl.sh "$1" 2>&1 | grep -v "^\[UART\]" | grep "cycles\|VERIFIED\|FAILED\|N=\|Assertion" > "results/$2.log"; echo "DONE" >> "results/$2.log"; }
for v in bench bench-nov32 bench-noskip bench-nocoalesce; do run $v-n$N.riscv $v-n$N & done
run bench-n4096.riscv bench-n4096 &
wait
