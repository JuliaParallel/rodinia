#include <stdio.h>
#include <omp.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;
    printf("num of threads = %d\n", omp_num_threads);
#ifdef OMP_OFFLOAD
#pragma omp target data map(a[:size*size])
#endif
    {
    for (i = 0; i < size; i++) {
        omp_set_num_threads(omp_num_threads);
#ifdef OMP_OFFLOAD
#pragma omp target teams
        {
#pragma omp distribute parallel for default(none) private(j, k, sum) shared(size, i, a)
#else
#pragma omp parallel for default(none) private(j, k, sum) shared(size, i, a)
#endif
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
#ifdef OMP_OFFLOAD
#pragma omp distribute parallel for default(none) private(j, k, sum) shared(size, i, a)
#else
#pragma omp parallel for default(none) private(j, k, sum) shared(size, i, a)
#endif
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
#ifdef OMP_OFFLOAD
        }
    }
#endif
    }
}
