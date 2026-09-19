// Generic host for one basic nn layer compiled through torch-mlir -> hwacha-mlir -> hwacha-cc.
// Reads the layer's input and PyTorch output from <layer>_check.bin, runs net(x), compares.
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

float *net(float *x);
static float xin[65536];

int main(void){
  const unsigned char *p=check_bin;
  int nin=*(const int*)p; p+=4; memcpy(xin,p,nin*4); p+=nin*4;
  int nout=*(const int*)p; p+=4;
  const float *ref=(const float*)p;
  atop=0;
  long c0=rdcycle();
  float *o=net(xin);
  long c1=rdcycle();
  float md=0,mx=0; int worst=0;
  for(int k=0;k<nout;k++){float d=fabsf(o[k]-ref[k]);if(d>md){md=d;worst=k;}if(fabsf(ref[k])>mx)mx=fabsf(ref[k]);}
  int ok=md<=1e-3f+1e-2f*mx;
  printf("%s: in %d out %d, max|diff|=%ld/1e6 max|ref|=%ld/1e6 %s, %ld cyc\n",
         MODEL,nin,nout,(long)(md*1e6f),(long)(mx*1e6f),ok?"ok":"FAIL",c1-c0);
  printf(ok?"%s PASS\n":"%s FAIL\n",MODEL);
  return !ok;
}
