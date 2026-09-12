#include "common.h"
#define NN 256
#define DEG 4
typedef struct { int starting; int no_of_edges; } Node;
void BFS_1_ct(long n, const Node *nodes, const int *edges, char *mask, char *umask, char *visited, int *cost, int no_of_nodes);
void BFS_2_ct(long n, char *mask, char *umask, char *visited, char *over, int no_of_nodes);
static Node nodes[NN]; static int edges[NN*DEG]; static char mask[NN], umask[NN], visited[NN]; static int cost[NN];
int main(void) {
  for (int i = 0; i < NN; i++) { nodes[i].starting = i*DEG; nodes[i].no_of_edges = DEG; for (int j = 0; j < DEG; j++) edges[i*DEG+j] = (j == 0) ? (i+1) % NN : (int)(rnd() % NN); }
  for (int i = 0; i < NN; i++) { mask[i] = 0; umask[i] = 0; visited[i] = 0; cost[i] = -1; }
  mask[0] = 1; visited[0] = 1; cost[0] = 0;
  char over;
  for (int it = 0; it < 4; it++) {
    printf("BFS_1 #%d start\n", it); BFS_1_ct(NN, nodes, edges, mask, umask, visited, cost, NN); printf("BFS_1 #%d done\n", it);
    over = 0;
    printf("BFS_2 #%d start\n", it); BFS_2_ct(NN, mask, umask, visited, &over, NN); printf("BFS_2 #%d done over=%d\n", it, over);
  }
  int n = 0; for (int i = 0; i < NN; i++) if (cost[i] >= 0) n++; printf("reached %d\n", n); return 0; }
