#!/bin/bash
# Build the Chipyard Verilator simulator of HwachaRocketConfig (detached: setsid nohup ./scripts-build-sim.sh &)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ~/hwacha-compiler/chipyard/.conda-env
export RISCV=~/hwacha-compiler/chipyard/.conda-env/esp-tools
cd ~/hwacha-compiler/chipyard/sims/verilator
source ../../env.sh > /dev/null 2>&1
make CONFIG=HwachaRocketConfig VERILATOR_THREADS=${THREADS:-8} SIM_OPT_CXXFLAGS=${OPT:--O2} -j2
echo "BUILD_EXIT=$?"
