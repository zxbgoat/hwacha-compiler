#!/bin/bash
# Run a benchmark binary on the Verilator Hwacha simulator: ./run-rtl.sh bench.riscv
SIM=$HOME/hwacha-compiler/chipyard/sims/verilator/simulator-chipyard.harness-HwachaRocketConfig
exec "$SIM" +permissive +max-cycles=500000000 +permissive-off "$@"
