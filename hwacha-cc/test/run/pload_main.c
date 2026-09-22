#include <stdio.h>
#include "util.h"
#define SX 32
#define SY 32
#define WIN 8
#define STRIDE 34
extern long hwacha_group_size;
void pload_ct(long n, const int *in, int *out, int sizeX, int sizeY, int nrows);
void pload0_ct(long n, const int *in, int *out, int sizeX, int sizeY, int nrows);
void pcol_ct(long n, const int *in, int *out, int sizeX, int sizeY, int group_y);
static int in[SX*SY], out[64*SX], ref[64*SX];
struct PIO { int CHECKED; int end, stride; }; struct PL { int CHECKED; int last; };
static void mirror(int *d, const int sizeD) { if (*d >= sizeD) *d = 2 * sizeD - 2 - *d; else if (*d < 0) *d = -*d; }
static int initialize_PixelIO(struct PIO *pIO, int CHECK, int sizeX, int sizeY, int firstX, int firstY) { pIO->CHECKED = CHECK; pIO->end = CHECK ? (sizeY * sizeX + firstX) : 0; pIO->stride = sizeX; return firstX + sizeX * firstY; }
static void init_PixelLoader(struct PL *l, int sizeX, int sizeY, int firstX, int firstY, struct PIO *pIO, int CHECK) { mirror(&firstX, sizeX); l->last = initialize_PixelIO(pIO, CHECK, sizeX, sizeY, firstX, firstY) - sizeX; }
static int loadFrom(struct PL *l, const int *in, struct PIO *pIO, int CHECK) { l->last += pIO->stride; if (CHECK && l->last == pIO->end) { l->last -= 2 * pIO->stride; pIO->stride = 0 - pIO->stride; } return in[l->last]; }
static void ref_pload0(int n) { for (int x = 0; x < SX; x++) { struct PL l; struct PIO io; init_PixelLoader(&l, SX, SY, x, 1, &io, 0); for (int i = 0; i < n; i++) ref[i*SX + x] = loadFrom(&l, in, &io, 0); } }
static void ref_pload(int n) { for (int x = 0; x < SX; x++) { struct PL l; struct PIO io; init_PixelLoader(&l, SX, SY, x, 0, &io, 1); for (int i = 0; i < n; i++) ref[i*SX + x] = loadFrom(&l, in, &io, 1); } }
static void ref_pcol(int group_y) { static int buf[(WIN+3)*STRIDE];
  for (int x = 0; x < SX; x++) { struct PL loader; struct PIO io; int p0, p1, p2, off = x; int firstY = group_y * WIN;
    if (group_y == 0) { init_PixelLoader(&loader, SX, SY, x, firstY, &io, 1); p2 = loadFrom(&loader, in, &io, 1); p1 = loadFrom(&loader, in, &io, 1); p0 = loadFrom(&loader, in, &io, 1); init_PixelLoader(&loader, SX, SY, x, firstY + 1, &io, 1); }
    else { init_PixelLoader(&loader, SX, SY, x, firstY - 2, &io, 1); p0 = loadFrom(&loader, in, &io, 1); p1 = loadFrom(&loader, in, &io, 1); p2 = loadFrom(&loader, in, &io, 1); }
    buf[off] = p0; buf[off + STRIDE] = p1; buf[off + 2*STRIDE] = p2;
    for (int i = 3; i < 3 + WIN; i++) buf[off + i*STRIDE] = loadFrom(&loader, in, &io, 1);
    p0 = buf[off + WIN*STRIDE]; p1 = buf[off + (WIN+1)*STRIDE]; p2 = buf[off + (WIN+2)*STRIDE];
    { int steps = (WIN + 2) / 2; for (int i = 0; i < steps; i++) { int row = 2*i + 1; int prev = buf[off + (row-1)*STRIDE], next = buf[off + (row+1)*STRIDE]; buf[off + row*STRIDE] -= (prev + next) / 2; } }
    { int steps = (WIN + 3) / 2 - 1; for (int i = 0; i < steps; i++) { int row = 2 + 2*i; int prev = buf[off + (row-1)*STRIDE], next = buf[off + (row+1)*STRIDE]; buf[off + row*STRIDE] += (prev + next + 2) / 4; } }
    for (int r = 2; r < 2 + WIN; r++) ref[(r-2)*SX + x] = buf[off + r*STRIDE];
    ref[WIN*SX + x] = p0; ref[(WIN+1)*SX + x] = p1; ref[(WIN+2)*SX + x] = p2; } }
static int check(const char *name, int rows) { int bad = 0; for (int i = 0; i < rows*SX; i++) if (out[i] != ref[i]) { if (bad < 4) printf("  %s[r%d c%d]: got %d want %d\n", name, i / SX, i % SX, out[i], ref[i]); bad++; }
  printf("%-18s %s (%d mismatches / %d)\n", name, bad ? "FAIL" : "PASS", bad, rows*SX); return bad != 0; }
int main(void) { int fail = 0; hwacha_group_size = SX;
  for (int i = 0; i < SX*SY; i++) in[i] = (i * 7919) % 251 - 100;
  for (int i = 0; i < 64*SX; i++) out[i] = -7777; ref_pload(40); pload_ct(SX, in, out, SX, SY, 40); fail |= check("pload", 40);
  for (int i = 0; i < 64*SX; i++) out[i] = -7777; ref_pload0(8); pload0_ct(SX, in, out, SX, SY, 8); fail |= check("pload0", 8);
  for (int g = 0; g < 3; g++) { static const char *nms[3] = {"pcol g0", "pcol g1", "pcol g2"}; const char *nm = nms[g]; for (int i = 0; i < 64*SX; i++) out[i] = -7777; ref_pcol(g); pcol_ct(SX, in, out, SX, SY, g); fail |= check(nm, WIN + 3); }
  printf("%s\n", fail ? "SOME KERNELS FAILED" : "ALL KERNELS PASSED"); return fail; }
