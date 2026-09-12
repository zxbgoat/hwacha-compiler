#include "riscv_test.h"
#include "test_macros.h"
#ifndef VLREQ
#define VLREQ 512
#endif
RVTEST_RV64UV
RVTEST_CODE_BEGIN
  vsetcfg 5,1
  li a4,VLREQ
  vsetvl a4,a4
  la a5,src
  vmca va1,a5
  la a6,dest
  vmca va2, a6
1:auipc a0,%pcrel_hi(vtcode)
  vf %pcrel_lo(1b)(a0)
  fence
  li a2,0          # index
  li a3,0          # mismatch count
  li a7,-1         # first mismatch index
loop:
  ld a0,0(a6)
  ld a1,0(a5)
  beq a0,a1,ok
  addi a3,a3,1
  bgez a7,ok
  mv a7,a2
ok:
  addi a6,a6,8
  addi a5,a5,8
  addi a2,a2,1
  bne a2,a4,loop
  # report: TESTNUM = (vl<<20) | (first<<10) | mismatches ; go to fail so tohost=(TESTNUM<<1)|1
  slli TESTNUM,a4,20
  slli a7,a7,10
  li t0,0xffc00
  and a7,a7,t0
  or TESTNUM,TESTNUM,a7
  or TESTNUM,TESTNUM,a3
  j fail
.align 3
vtcode:
  vpset vp0
  vld vv1, va1
  vsd vv1, va2
  vstop
  TEST_PASSFAIL
RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
  TEST_DATA
src:
#include "data_d.h"
dest:
  .skip 16384
RVTEST_DATA_END
