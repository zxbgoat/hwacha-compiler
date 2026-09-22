// minimal repro of dwt2d's column loader / vertical lifting pattern: per-lane private structs
// (SROA'd into loop-carried values), a mirrored pixel loader, a __local column buffer
struct PIO { int CHECKED; int end, stride; };
struct PL { int CHECKED; int last; };
static void mirror(int *d, const int sizeD) { if (*d >= sizeD) *d = 2 * sizeD - 2 - *d; else if (*d < 0) *d = -*d; }
static int initialize_PixelIO(struct PIO *pIO, int CHECK, int sizeX, int sizeY, int firstX, int firstY) {
  pIO->CHECKED = CHECK; pIO->end = CHECK ? (sizeY * sizeX + firstX) : 0; pIO->stride = sizeX; return firstX + sizeX * firstY; }
static void init_PixelLoader(struct PL *l, int sizeX, int sizeY, int firstX, int firstY, struct PIO *pIO, int CHECK) {
  mirror(&firstX, sizeX); l->last = initialize_PixelIO(pIO, CHECK, sizeX, sizeY, firstX, firstY) - sizeX; }
static int loadFrom(struct PL *l, __global const int *in, struct PIO *pIO, int CHECK) {
  l->last += pIO->stride;
  if (CHECK && l->last == pIO->end) { l->last -= 2 * pIO->stride; pIO->stride = 0 - pIO->stride; }
  return in[l->last]; }

__kernel void pload(__global const int *in, __global int *out, int sizeX, int sizeY, int n) {
  int x = get_global_id(0);
  struct PL l; struct PIO io;
  init_PixelLoader(&l, sizeX, sizeY, x, 0, &io, 1);
  for (int i = 0; i < n; i++) out[i * sizeX + x] = loadFrom(&l, in, &io, 1);
}

struct Col { int CHECKED; struct PL loader; int offset; int p0, p1, p2; };
#define WIN 8
#define STRIDE 34
__kernel void pcol(__global const int *in, __global int *out, int sizeX, int sizeY, int group_y) {
  __local int buf[(WIN + 3) * STRIDE];
  int x = get_local_id(0);
  struct Col c; struct PIO io;
  c.offset = x;
  int firstY = group_y * WIN;
  if (group_y == 0) {
    init_PixelLoader(&c.loader, sizeX, sizeY, x, firstY, &io, 1);
    c.p2 = loadFrom(&c.loader, in, &io, 1); c.p1 = loadFrom(&c.loader, in, &io, 1); c.p0 = loadFrom(&c.loader, in, &io, 1);
    init_PixelLoader(&c.loader, sizeX, sizeY, x, firstY + 1, &io, 1);
  } else {
    init_PixelLoader(&c.loader, sizeX, sizeY, x, firstY - 2, &io, 1);
    c.p0 = loadFrom(&c.loader, in, &io, 1); c.p1 = loadFrom(&c.loader, in, &io, 1); c.p2 = loadFrom(&c.loader, in, &io, 1);
  }
  buf[c.offset + 0 * STRIDE] = c.p0; buf[c.offset + 1 * STRIDE] = c.p1; buf[c.offset + 2 * STRIDE] = c.p2;
  for (int i = 3; i < 3 + WIN; i++) buf[c.offset + i * STRIDE] = loadFrom(&c.loader, in, &io, 1);
  c.p0 = buf[c.offset + (WIN + 0) * STRIDE]; c.p1 = buf[c.offset + (WIN + 1) * STRIDE]; c.p2 = buf[c.offset + (WIN + 2) * STRIDE];
  { int steps = (WIN + 3 - 1) / 2;
    for (int i = 0; i < steps; i++) { int row = 2 * i + 1; int prev = buf[c.offset + (row - 1) * STRIDE], next = buf[c.offset + (row + 1) * STRIDE]; buf[c.offset + row * STRIDE] -= (prev + next) / 2; } }
  { int steps = (WIN + 3) / 2 - 1;
    for (int i = 0; i < steps; i++) { int row = 2 + 2 * i; int prev = buf[c.offset + (row - 1) * STRIDE], next = buf[c.offset + (row + 1) * STRIDE]; buf[c.offset + row * STRIDE] += (prev + next + 2) / 4; } }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int r = 2; r < 2 + WIN; r++) out[(r - 2) * sizeX + x] = buf[c.offset + r * STRIDE];
  out[WIN * sizeX + x] = c.p0; out[(WIN + 1) * sizeX + x] = c.p1; out[(WIN + 2) * sizeX + x] = c.p2;
}

// CHECK == 0: no mirroring branch, so the load address is a clean add-recurrence {in + 4*(x + sizeX), +, 4*sizeX}
__kernel void pload0(__global const int *in, __global int *out, int sizeX, int sizeY, int n) {
  int x = get_global_id(0);
  struct PL l; struct PIO io;
  init_PixelLoader(&l, sizeX, sizeY, x, 1, &io, 0);
  for (int i = 0; i < n; i++) out[i * sizeX + x] = loadFrom(&l, in, &io, 0);
}
