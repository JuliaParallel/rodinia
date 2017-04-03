#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <matrix.h>
#undef catch
#undef min
#undef max

#define PI 3.14159


extern int double_eq(double f1, double f2);
extern MAT *gradient_x(MAT *input);
extern MAT *gradient_y(MAT *input);
extern double mean(VEC *in);
extern double std_dev(VEC *in);

#ifdef __cplusplus
}
#endif
