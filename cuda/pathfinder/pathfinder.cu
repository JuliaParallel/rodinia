#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "../../common/cuda/profile_main.h"

#define BLOCK_SIZE 256
#define HALO 1  // halo width along one direction when advancing to the next iteration

int rows, cols;
int *data;
int **wall;
int *result;
int pyramid_height;

void init(int argc, char **argv) {
    if (argc == 4) {
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
        pyramid_height = atoi(argv[3]);
    } else {
        printf("Usage: dynproc row_len col_len pyramid_height\n");
        exit(1);
    }
    data = new int[rows * cols];
    wall = new int *[rows];
    for (int n = 0; n < rows; n++)
        wall[n] = data + cols * n;
    result = new int[cols];

    srand(7);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            wall[i][j] = rand() % 10;
        }
    }

    if (getenv("OUTPUT")) {
        FILE *file = fopen("output.txt", "w");

        fprintf(file, "wall:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(file, "%d ", wall[i][j]);
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc,
                               int *gpuResults, int cols, int rows,
                               int startStep, int border) {

    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // each block finally computes result for a small block
    // after N iterations.
    // it is the non-overlapping small blocks that cover
    // all the input data

    // calculate the small block size
    int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

    // calculate the boundary for the block according to
    // the boundary of its small block
    int blkX = small_block_cols * bx - border;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    // calculate the global thread coordination
    int xidx = blkX + tx;

    // effective range within this block that falls within
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1)
                                         : BLOCK_SIZE - 1;

    int W = tx - 1;
    int E = tx + 1;

    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if (IN_RANGE(xidx, 0, cols - 1)) {
        prev[tx] = gpuSrc[xidx];
    }

    __syncthreads();

    bool computed;
    for (int i = 0; i < iteration; i++) {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
            computed = true;
            int left = prev[W];
            int up = prev[tx];
            int right = prev[E];
            int shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            int index = cols * (startStep + i) + xidx;
            result[tx] = shortest + gpuWall[index];
        }
        __syncthreads();
        if (i == iteration - 1)
            break;
        if (computed) // Assign the computation range
            prev[tx] = result[tx];
        __syncthreads();
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the
    // small block perform the calculation and switch on ``computed''
    if (computed) {
        gpuResults[xidx] = result[tx];
    }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
              int pyramid_height, int blockCols, int borderCols) {
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    int src = 1, dst = 0;
    for (int t = 0; t < rows - 1; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        PROFILE((
            dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
                gpuResult[dst], cols, rows, t, borderCols)
        ));
    }
    return dst;
}

int run(int argc, char **argv) {
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
    int blockCols =
        cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: "
           "%d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
           pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols,
           smallBlockCol);

    int *gpuWall, *gpuResult[2];
    int size = rows * cols;

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    cudaMalloc((void **)&gpuResult[0], sizeof(int) * cols);
    cudaMalloc((void **)&gpuResult[1], sizeof(int) * cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&gpuWall, sizeof(int) * (size - cols));
    cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols),
               cudaMemcpyHostToDevice);


    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height,
                              blockCols, borderCols);

    cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols,
               cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed = (end.tv_sec - start.tv_sec)
        + (end.tv_nsec - start.tv_nsec)/1E9;
    printf("%.6f seconds\n", elapsed);

    if (getenv("OUTPUT")) {
        FILE *file = fopen("output.txt", "a");

        fprintf(file, "data:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", data[i]);
        fprintf(file, "\n");

        fprintf(file, "result:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", result[i]);
        fprintf(file, "\n");

        fclose(file);
    }

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete[] data;
    delete[] wall;
    delete[] result;

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    if (getenv("PROFILE"))
        profile_start();

    run(argc, argv);

    if (getenv("PROFILE"))
        profile_stop();

    return EXIT_SUCCESS;
}

