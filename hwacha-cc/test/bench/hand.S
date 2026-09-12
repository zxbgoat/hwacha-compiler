// Hand translation of kernels.cl into Hwacha v4 code, written the way a
// compiler would emit it.  Each kernel produces two pieces:
//   <name>_ct : control-thread function (RISC-V + Hwacha control insns),
//               C-callable, does vsetcfg / stripmine / vmca / vmcs / vf
//   <name>_wt : worker-thread vector-fetch block, ends with vstop
//
// Mapping rules used (the "spec" for a future compiler):
//  R1 every __global pointer indexed by get_global_id(0) with a constant
//     stride becomes a va register: base advanced by vl*stride per stripmine
//  R2 every uniform kernel argument becomes a vs register (vmcs once, outside
//     the loop)
//  R3 get_global_id(0) used as data = vs_offset + veidx, where vs_offset is
//     re-sent (vmcs) every stripmine iteration
//  R4 if/else on a per-element condition -> vcmp* into a vp register, then
//     both arms predicated with @vp / @!vp (no vcjal needed unless the arm is
//     long enough to be worth a consensual skip)
//  R5 vsetcfg counts registers actually used: (#v64, #v32, #v16, #vp)

#include "vec-util.h"

.text
.align 2

// ---------------------------------------------------------------- K1 saxpy
// void hand_saxpy_ct(long n, float a, const float *x, float *y)
//   a0=n  fa0=a  a1=x  a2=y
.globl hand_saxpy_ct
.type  hand_saxpy_ct,@function
hand_saxpy_ct:
    li   t0, VCFG(0, 2, 0, 1)      // R5: vv0,vv1 as 32-bit, vp0
    vsetcfg t0
    fmv.x.s t1, fa0
    vmcs vs1, t1                    // R2: a -> vs1
1:  vsetvl t0, a0
    vmca va0, a1                    // R1: x -> va0
    vmca va1, a2                    // R1: y -> va1
    la   t2, hand_saxpy_wt
    vf   0(t2)
    slli t1, t0, 2                  // advance by vl * sizeof(float)
    add  a1, a1, t1
    add  a2, a2, t1
    sub  a0, a0, t0
    bnez a0, 1b
    fence
    ret

.align 3
hand_saxpy_wt:
    vpset vp0
    vlw  vv0, va0                   // x[i]
    vlw  vv1, va1                   // y[i]
    vfmadd.s vv1, vs1, vv0, vv1     // a*x[i] + y[i]
    vsw  vv1, va1
    vstop

// ---------------------------------------------------------- K2 clamp_scale
// void hand_clamp_scale_ct(long n, float a, const float *x, float *y)
.globl hand_clamp_scale_ct
.type  hand_clamp_scale_ct,@function
hand_clamp_scale_ct:
    li   t0, VCFG(0, 2, 0, 2)      // vv0,vv1 (32-bit), vp0,vp1
    vsetcfg t0
    fmv.x.s t1, fa0
    vmcs vs1, t1
1:  vsetvl t0, a0
    vmca va0, a1
    vmca va1, a2
    la   t2, hand_clamp_scale_wt
    vf   0(t2)
    slli t1, t0, 2
    add  a1, a1, t1
    add  a2, a2, t1
    sub  a0, a0, t0
    bnez a0, 1b
    fence
    ret

.align 3
hand_clamp_scale_wt:
    vpset vp0
    vlw  vv0, va0                   // v = x[i]
    vcmpflt.s vp1, vs0, vv0         // R4: vp1 = (0.0f < v)   (vs0 == 0 == +0.0f)
@vp1  vfmul.s vv1, vv0, vs1         // then-arm: y = a*v
@!vp1 vadd    vv1, vs0, vs0         // else-arm: y = 0
    vsw  vv1, va1
    vstop

// ----------------------------------------------------------------- K3 iota
// void hand_iota_ct(long n, long base, long *c)
//   a0=n a1=base a2=c
.globl hand_iota_ct
.type  hand_iota_ct,@function
hand_iota_ct:
    li   t0, VCFG(1, 0, 0, 1)      // vv0 (64-bit), vp0
    vsetcfg t0
    vmcs vs1, a1                    // R2: base -> vs1
    li   t3, 0                      // running global-id offset
1:  vsetvl t0, a0
    vmcs vs2, t3                    // R3: offset of this stripmine -> vs2
    vmca va0, a2
    la   t2, hand_iota_wt
    vf   0(t2)
    slli t1, t0, 3
    add  a2, a2, t1
    add  t3, t3, t0
    sub  a0, a0, t0
    bnez a0, 1b
    fence
    ret

.align 3
hand_iota_wt:
    vpset vp0
    veidx vv0                       // element index within this stripmine
    vadd  vv0, vv0, vs2             // R3: i = offset + eidx
    vadd  vv0, vv0, vv0             // 2*i  (immediate forms vaddi/vslli are vs-only)
    vadd  vv0, vv0, vs1             // base + 2*i
    vsd   vv0, va0
    vstop
