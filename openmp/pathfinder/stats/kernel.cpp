        for (int n = 0; n < cols; n++) {
            min = src[n];
            if (n > 0)
                min = MIN(min, src[n - 1]);
            if (n < cols - 1)
                min = MIN(min, src[n + 1]);
            dst[n] = wall[t + 1][n] + min;
        }
