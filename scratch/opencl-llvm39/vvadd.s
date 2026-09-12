	.text
	.file	"vvadd.ll"
	.p2align	2
	.type	vvadd,@function
vvadd:                                  # @vvadd
	.cfi_startproc
# BB#0:                                 # %entry
	vpset	vp0
	veidx	vv0
	ld	x5, 0(x11)
	ld	x6, 0(x10)
	vadd	vs0, x6, x5
	sd	vs0, 0(x12)
	vstop
Lfunc_end0:
	.size	vvadd, Lfunc_end0-vvadd
	.cfi_endproc

	.globl	host
	.p2align	2
	.type	host,@function
host:                                   # @host
	.cfi_startproc
# BB#0:                                 # %entry
	addi	x2, x2, 0
Ltmp0:
	.cfi_def_cfa_offset 0
	addi	x10, x10, 8
	addi	x11, x11, 8
	addi	x12, x12, 8
	lui	x5, %hi(vvadd)
	addi	x5, x5, %lo(vvadd)
	vsetcfg	x6,1,0,0,1
	li	x6, 4
	vsetvl	x6,x6
	vf	0(x5)
	ret
Lfunc_end1:
	.size	host, Lfunc_end1-host
	.cfi_endproc


