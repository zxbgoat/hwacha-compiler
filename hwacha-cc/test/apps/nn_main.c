// Rodinia nn: distance from every record to a query point
#include "common.h"
#include <math.h>
#define N 2048
typedef struct { float lat, lng; } LatLong;
void NearestNeighbor_ct(long n, LatLong *loc, float *dist, int numRecords, float lat, float lng);
static LatLong loc[N]; static float dist[N], ref[N];
int main(void) {
  const float lat = 30.0f, lng = -90.0f;
  for (int i = 0; i < N; i++) { loc[i].lat = frand(0, 90); loc[i].lng = frand(-180, 180); dist[i] = -1; }
  unsigned long c0 = cyc();
  for (int i = 0; i < N; i++) ref[i] = sqrtf((lat-loc[i].lat)*(lat-loc[i].lat) + (lng-loc[i].lng)*(lng-loc[i].lng));
  unsigned long c1 = cyc();
  REPORT("nn scalar", c0, c1, N);
  c0 = cyc(); NearestNeighbor_ct(N, loc, dist, N, lat, lng); c1 = cyc();
  REPORT("nn hwacha-cc", c0, c1, N);
  int bad = 0; for (int i = 0; i < N; i++) { float d = dist[i] - ref[i]; if (d < 0) d = -d; if (d > 1e-3f * (ref[i] + 1)) bad++; }
  printf("nn %s (%d mismatches / %d)\n", bad ? "FAIL" : "PASS", bad, N); return bad != 0;
}
