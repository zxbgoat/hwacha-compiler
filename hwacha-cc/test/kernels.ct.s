	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0"
	.file	"hwacha-ct"
	.text
	.globl	saxpy_ct                        # -- Begin function saxpy_ct
	.p2align	2
	.type	saxpy_ct,@function
saxpy_ct:                               # @saxpy_ct
	.cfi_startproc
# %bb.0:                                # %entry
	fmv.x.w	a3, fa0
	li	a4, -1
	slli	a4, a4, 63
	slli	a3, a3, 32
	addi	a4, a4, 515
	srli	a3, a3, 32
	#APP
	vsetcfg a4
	#NO_APP
	#APP
	vmcs vs1, a3
	#NO_APP
	beqz	a0, .LBB0_3
# %bb.1:                                # %stripmine.preheader
	lui	a3, %hi(saxpy_wt)
	addi	a3, a3, %lo(saxpy_wt)
.LBB0_2:                                # %stripmine
                                        # =>This Inner Loop Header: Depth=1
	#APP
	vsetvl a4, a0
	#NO_APP
	#APP
	vmca va0, a1
	#NO_APP
	#APP
	vmca va1, a2
	#NO_APP
	#APP
	vf 0(a3)
	#NO_APP
	sub	a0, a0, a4
	slli	a4, a4, 2
	add	a1, a1, a4
	add	a2, a2, a4
	bnez	a0, .LBB0_2
.LBB0_3:                                # %done
	#APP
	fence
	#NO_APP
	ret
.Lfunc_end0:
	.size	saxpy_ct, .Lfunc_end0-saxpy_ct
	.cfi_endproc
                                        # -- End function
	.globl	clamp_scale_ct                  # -- Begin function clamp_scale_ct
	.p2align	2
	.type	clamp_scale_ct,@function
clamp_scale_ct:                         # @clamp_scale_ct
	.cfi_startproc
# %bb.0:                                # %entry
	fmv.x.w	a3, fa0
	li	a4, -1
	slli	a4, a4, 63
	slli	a3, a3, 32
	addi	a4, a4, 1026
	srli	a3, a3, 32
	#APP
	vsetcfg a4
	#NO_APP
	#APP
	vmcs vs1, a3
	#NO_APP
	beqz	a0, .LBB1_3
# %bb.1:                                # %stripmine.preheader
	lui	a3, %hi(clamp_scale_wt)
	addi	a3, a3, %lo(clamp_scale_wt)
.LBB1_2:                                # %stripmine
                                        # =>This Inner Loop Header: Depth=1
	#APP
	vsetvl a4, a0
	#NO_APP
	#APP
	vmca va0, a1
	#NO_APP
	#APP
	vmca va1, a2
	#NO_APP
	#APP
	vf 0(a3)
	#NO_APP
	sub	a0, a0, a4
	slli	a4, a4, 2
	add	a1, a1, a4
	add	a2, a2, a4
	bnez	a0, .LBB1_2
.LBB1_3:                                # %done
	#APP
	fence
	#NO_APP
	ret
.Lfunc_end1:
	.size	clamp_scale_ct, .Lfunc_end1-clamp_scale_ct
	.cfi_endproc
                                        # -- End function
	.globl	iota_ct                         # -- Begin function iota_ct
	.p2align	2
	.type	iota_ct,@function
iota_ct:                                # @iota_ct
	.cfi_startproc
# %bb.0:                                # %entry
	li	a3, -1
	slli	a3, a3, 63
	addi	a3, a3, 514
	#APP
	vsetcfg a3
	#NO_APP
	li	a3, 1
	#APP
	vmcs vs2, a3
	#NO_APP
	#APP
	vmcs vs3, a1
	#NO_APP
	beqz	a0, .LBB2_3
# %bb.1:                                # %stripmine.preheader
	li	a1, 0
	lui	a3, %hi(iota_wt)
	addi	a3, a3, %lo(iota_wt)
.LBB2_2:                                # %stripmine
                                        # =>This Inner Loop Header: Depth=1
	#APP
	vsetvl a4, a0
	#NO_APP
	#APP
	vmca va0, a2
	#NO_APP
	#APP
	vmcs vs1, a1
	#NO_APP
	sub	a0, a0, a4
	add	a1, a1, a4
	slli	a4, a4, 3
	#APP
	vf 0(a3)
	#NO_APP
	add	a2, a2, a4
	bnez	a0, .LBB2_2
.LBB2_3:                                # %done
	#APP
	fence
	#NO_APP
	ret
.Lfunc_end2:
	.size	iota_ct, .Lfunc_end2-iota_ct
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
