// Host for a minimal DiT compiled through torch-mlir -> hwacha-mlir -> hwacha-cc. The model returns the
// per-token output [1, nt, P*P*C]; the trivial unpatchify (a fixed pixel permutation) is left to the host.
// Compares one forward against the PyTorch tokens exported next to the model (dit_check.bin).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
asm(".section .rodata\n.balign 8\n.globl check_bin\ncheck_bin:\n.incbin \"" CHECK_FILE "\"\n.previous");
extern const unsigned char check_bin[];
static long rdcycle(void){long c;asm volatile("rdcycle %0":"=r"(c));return c;}
float __math_oflowf(unsigned s){return s?-1.0f/0.0f:1.0f/0.0f;}
float __math_uflowf(unsigned s){return s?-0.0f:0.0f;}
int *__errno(void){static int e;return &e;}
uintptr_t handle_trap(uintptr_t c,uintptr_t e,uintptr_t r[32]){uintptr_t t;asm volatile("csrr %0, mtval":"=r"(t));printf("TRAP %lx epc %lx tval %lx\n",c,e,t);exit(1);}
#define ARENA (1<<24)
static char arena[ARENA] __attribute__((aligned(64)));static long atop;
void *malloc(size_t n){long p=(atop+63)&~63L;atop=p+n;if(atop>ARENA){printf("arena of\n");exit(1);}return arena+p;}
void free(void*p){(void)p;}
void *aligned_alloc(size_t a,size_t n){(void)a;return malloc(n);}

float *dit(float *x, float *t, long *y);
static float xin[8192], tin[1];
static long yin[1];

int main(void){
  const unsigned char *p=check_bin;
  int nx=*(const int*)p; p+=4;
  memcpy(xin,p,nx*4); p+=nx*4;
  float tv; memcpy(&tv,p,4); p+=4; tin[0]=tv;
  int cls=*(const int*)p; p+=4; yin[0]=cls;
  int ntok=*(const int*)p; p+=4;
  const float *ref=(const float*)p;
  printf("DiT: t*1e3=%ld class %d, %d tokens\n",(long)(tv*1e3f),cls,ntok);
  atop=0;
  long c0=rdcycle();
  float *o=dit(xin,tin,yin);
  long c1=rdcycle();
  float md=0,mx=0; int worst=0;
  for(int k=0;k<ntok;k++){float d=fabsf(o[k]-ref[k]);if(d>md){md=d;worst=k;}if(fabsf(ref[k])>mx)mx=fabsf(ref[k]);}
  int ok=md<=1e-3f+2e-2f*mx;
  printf("forward vs PyTorch: max|diff|=%ld/1e6 max|ref|=%ld/1e6 worst[%d] hw=%ld/1e6 ref=%ld/1e6 %s; %ld cyc, arena %ldKB\n",
         (long)(md*1e6f),(long)(mx*1e6f),worst,(long)(o[worst]*1e6f),(long)(ref[worst]*1e6f),ok?"ok":"FAIL",c1-c0,atop/1024);
  printf(ok?"dit PASS\n":"dit FAIL\n");
  return !ok;
}
