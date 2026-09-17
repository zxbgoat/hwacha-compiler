// Host for a torchvision classification model (torch-mlir -> hwacha-mlir -> hwacha-cc).
// Input, arity and a PyTorch reference come from <model>_check.bin (export_tv.py). Entry = `net`.
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
uintptr_t handle_trap(uintptr_t c,uintptr_t e,uintptr_t r[32]){uintptr_t t;asm volatile("csrr %0, mtval":"=r"(t));printf("TRAP cause %lx epc %lx tval %lx\n",c,e,t);exit(1);}
#define ARENA (1<<26)
static char arena[ARENA] __attribute__((aligned(64))); static long atop;
void *malloc(size_t n){long p=(atop+63)&~63L;atop=p+n;if(atop>ARENA){printf("arena overflow\n");exit(1);}return arena+p;}
void free(void*p){(void)p;}
void *aligned_alloc(size_t a,size_t n){(void)a;return malloc(n);}
static const char *ff(float v){static char b[8][32];static int k;char*o=b[k++&7],*q=o;long m=(long)(fabsf(v)*1e6f+0.5f),ip=m/1000000,fp=m%1000000;if(v<0)*q++='-';char t[24];int n=0;do{t[n++]='0'+ip%10;ip/=10;}while(ip);while(n)*q++=t[--n];*q++='.';for(long d=100000;d;d/=10)*q++='0'+(fp/d)%10;*q=0;return o;}
float *net(float *);
static float in[3*64*64];
int main(void){
  const unsigned char *p=check_bin; int nin=*(const int*)p; p+=4;
  int isz=*(const int*)p; p+=4; memcpy(in,p,isz*sizeof(float)); p+=isz*sizeof(float);
  int nout=*(const int*)p; p+=4; const float *ref=(const float*)p;
  (void)nin;
  long c0=rdcycle(); float *y=net(in); long c1=rdcycle();
  float md=0,mx=0; int am=0; for(int k=0;k<nout;k++){float d=fabsf(y[k]-ref[k]);if(d>md)md=d;if(fabsf(ref[k])>mx)mx=fabsf(ref[k]);if(y[k]>y[am])am=k;}
  int ra=0; for(int k=0;k<nout;k++) if(ref[k]>ref[ra]) ra=k;
  // PyTorch's own forward on random weights: certify Hwacha reproduces it (numeric agreement). argmax
  // is reported for information -- with random (untrained) weights the logits can be near-degenerate,
  // so an argmax tie is not a computation error as long as the outputs match numerically.
  int ok = (md <= 1e-3f + 1e-2f * mx) && (am == ra);
  printf("%s: %d classes, %ld cycles; max|diff|=%s max|ref|=%s; argmax hw=%d ref=%d%s; %s\n", MODEL, nout, c1-c0, ff(md), ff(mx), am, ra, am==ra?" (match)":"", ok?"ok":"FAIL");
  printf(ok?"tv PASS\n":"tv FAIL\n");
  return !ok;
}
