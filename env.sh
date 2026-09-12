# source this file: sets up the Hwacha toolchain built in this directory
export HWACHA_ROOT="$HOME/hwacha-compiler"
export PATH="$HWACHA_ROOT/install/bin:$PATH"
# spike-hlog: same Spike but prints "H:" Hwacha commit-log lines to stderr
alias spike-hlog="$HWACHA_ROOT/install-hlog/bin/spike"
