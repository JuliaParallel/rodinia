diff --git a/openmp/pathfinder/pathfinder.cpp b/openmp/pathfinder/pathfinder.cpp
index e5bfcba..2351439 100644
--- a/openmp/pathfinder/pathfinder.cpp
+++ b/openmp/pathfinder/pathfinder.cpp
@@ -78,11 +78,22 @@ void run(int argc, char **argv) {
    src = new int[cols];

    pin_stats_reset();
{+#ifdef OMP_OFFLOAD+}
{+#pragma omp target enter data map (to: src[:cols], wall[:rows], dst[:cols])+}
{+    for (int i = 0; i < rows; i++) {+}
{+      #pragma omp target enter data map(to: wall[i][:cols])+}
{+    }+}
{+#endif+}
{+#endif+}
    for (int t = 0; t < rows - 1; t++) {
        temp = src;
        src = dst;
        dst = temp;
{+#ifdef OMP_OFFLOAD+}
{+#pragma omp target teams distribute parallel for private(min)+}
{+#else+}
#pragma omp parallel for private(min)
{+#endif+}
        for (int n = 0; n < cols; n++) {
            min = src[n];
            if (n > 0)
@@ -92,6 +103,10 @@ void run(int argc, char **argv) {
            dst[n] = wall[t + 1][n] + min;
        }
    }
{+#ifdef OMP_OFFLOAD+}
{+    // retrieve data+}
{+#pragma omp target exit data map (from: dst[:cols])+}
{+#endif+}

    pin_stats_pause(cycles);
    pin_stats_dump(cycles);
