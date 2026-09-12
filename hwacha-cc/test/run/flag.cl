// minimal repro of pathfinder's "computed" flag: reset every iteration, set under a divergent condition,
// loop left through a break, flag read after the loop
__kernel void flagtest(__global const int *x, __global int *out, int iters) {
  int tx = get_local_id(0); int gid = get_global_id(0);
  bool computed;
  for (int i = 0; i < iters; i++) {
    computed = false;
    if (tx >= i + 1 && tx <= 100) computed = true;
    if (i == iters - 1) break;
  }
  if (computed) out[gid] = x[gid] + 1;
}
__kernel void flag_noloop(__global const int *x, __global int *out) {
  int tx = get_local_id(0); int gid = get_global_id(0);
  bool computed = false;
  if (tx >= 1 && tx <= 100) computed = true;
  if (computed) out[gid] = x[gid] + 1;
}
__kernel void flag_loop_nobreak(__global const int *x, __global int *out, int iters) {
  int tx = get_local_id(0); int gid = get_global_id(0);
  bool computed = false;
  for (int i = 0; i < iters; i++) { computed = false; if (tx >= i + 1 && tx <= 100) computed = true; }
  if (computed) out[gid] = x[gid] + 1;
}
