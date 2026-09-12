// Rodinia bfs: level-synchronous BFS with the two kernels, iterated from the host like the original
#include "common.h"
#ifndef NN
#define NN 2048
#endif
#define DEG 4
typedef struct { int starting; int no_of_edges; } Node;
void BFS_1_ct(long n, const Node *nodes, const int *edges, char *mask, char *umask, char *visited, int *cost, int no_of_nodes);
void BFS_2_ct(long n, char *mask, char *umask, char *visited, char *over, int no_of_nodes);
static Node nodes[NN]; static int edges[NN*DEG]; static char mask[NN], umask[NN], visited[NN]; static int cost[NN], ref[NN];
int main(void) {
  for (int i = 0; i < NN; i++) { nodes[i].starting = i*DEG; nodes[i].no_of_edges = DEG; for (int j = 0; j < DEG; j++) edges[i*DEG+j] = (j == 0) ? (i+1) % NN : (int)(rnd() % NN); }
  /* reference BFS from node 0 */
  unsigned long c0 = cyc();
  for (int i = 0; i < NN; i++) ref[i] = -1;
  { static int q[NN]; int h = 0, t = 0; ref[0] = 0; q[t++] = 0;
    while (h < t) { int u = q[h++]; for (int j = 0; j < DEG; j++) { int v = edges[u*DEG+j]; if (ref[v] < 0) { ref[v] = ref[u] + 1; q[t++] = v; } } } }
  unsigned long c1 = cyc(); REPORT("bfs scalar", c0, c1, NN);
  for (int i = 0; i < NN; i++) { mask[i] = 0; umask[i] = 0; visited[i] = 0; cost[i] = -1; }
  mask[0] = 1; visited[0] = 1; cost[0] = 0;
  char over; int iters = 0;
  c0 = cyc();
  do { over = 0; BFS_1_ct(NN, nodes, edges, mask, umask, visited, cost, NN); BFS_2_ct(NN, mask, umask, visited, &over, NN); iters++; } while (over && iters < 100);
  c1 = cyc(); REPORT("bfs hwacha-cc", c0, c1, NN);
  int bad = 0; for (int i = 0; i < NN; i++) if (cost[i] != ref[i]) bad++;
  printf("bfs %s (%d mismatches / %d, %d levels)\n", bad ? "FAIL" : "PASS", bad, NN, iters); return bad != 0;
}
