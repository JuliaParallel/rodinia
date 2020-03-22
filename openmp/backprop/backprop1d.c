/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */

#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include "backprop1d.h"
#include <math.h>


#define ABS(x) (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to, from, len)                                                \
    {                                                                          \
        register char *_to, *_from;                                            \
        register int _i, _l;                                                   \
        _to = (char *)(to);                                                    \
        _from = (char *)(from);                                                \
        _l = (len);                                                            \
        for (_i = 0; _i < _l; _i++)                                            \
            *_to++ = *_from++;                                                 \
    }

/*** Return random number between 0.0 and 1.0 ***/
float drnd() { return ((float)rand() / (float)BIGRND); }

/*** Return random number between -1.0 and 1.0 ***/
float dpn1() { return ((drnd() * 2.0) - 1.0); }

/*** The squashing function.  Currently, it's a sigmoid. ***/

float squash(x) float x;
{
    float m;
    // x = -x;
    // m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
    // return(1.0 / (1.0 + m));
    return (1.0 / (1.0 + exp(-x)));
}


/*** Allocate 1d array of floats ***/

float *alloc_1d_dbl(n) int n;
{
    float *new;

    new = (float *)malloc((unsigned)(n * sizeof(float)));
    if (new == NULL) {
        printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
        return (NULL);
    }
    return (new);
}


/*** Allocate 2d array of floats ***/

float **alloc_2d_dbl(m, n) int m, n;
{
    int i;
    float **new;

    new = (float **)malloc((unsigned)(m * sizeof(float *)));
    if (new == NULL) {
        printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
        return (NULL);
    }

    for (i = 0; i < m; i++) {
        new[i] = alloc_1d_dbl(n);
    }

    return (new);
}


void bpnn_randomize_weights(w, m, n) float *w;
int m, n;
{
    int i, j;

    for (i = 0; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            w[i*(n+1)+j] = (float)rand() / RAND_MAX;
            //  w[i][j] = dpn1();
        }
    }
}

void bpnn_randomize_row(w, m) float *w;
int m;
{
    int i;
    for (i = 0; i <= m; i++) {
        // w[i] = (float) rand()/RAND_MAX;
        w[i] = 0.1;
    }
}


void bpnn_zero_weights(w, m, n) float *w;
int m, n;
{
    int i, j;

    for (i = 0; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            w[i*(n+1)+j] = 0.0;
        }
    }
}


void bpnn_initialize(seed) int seed;
{
    printf("Random number generator seed: %d\n", seed);
    srand(seed);
}


BPNN *bpnn_internal_create(n_in, n_hidden, n_out) int n_in, n_hidden, n_out;
{
    BPNN *newnet;

    newnet = (BPNN *)malloc(sizeof(BPNN));
    if (newnet == NULL) {
        printf("BPNN_CREATE: Couldn't allocate neural network\n");
        return (NULL);
    }

    newnet->input_n = n_in;
    newnet->hidden_n = n_hidden;
    newnet->output_n = n_out;
    newnet->input_units = alloc_1d_dbl(n_in + 1);
    newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
    newnet->output_units = alloc_1d_dbl(n_out + 1);

    newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
    newnet->output_delta = alloc_1d_dbl(n_out + 1);
    newnet->target = alloc_1d_dbl(n_out + 1);

    newnet->input_weights = alloc_1d_dbl((n_in + 1)*(n_hidden + 1));
    newnet->hidden_weights = alloc_1d_dbl((n_hidden + 1)*(n_out + 1));

    newnet->input_prev_weights = alloc_1d_dbl((n_in + 1)*(n_hidden + 1));
    newnet->hidden_prev_weights = alloc_1d_dbl((n_hidden + 1)*( n_out + 1));

    return (newnet);
}


void bpnn_free(net) BPNN *net;
{
    int n1, n2, i;

    n1 = net->input_n;
    n2 = net->hidden_n;

    free((char *)net->input_units);
    free((char *)net->hidden_units);
    free((char *)net->output_units);

    free((char *)net->hidden_delta);
    free((char *)net->output_delta);
    free((char *)net->target);

    free((char *)net->input_weights);
    free((char *)net->input_prev_weights);

    free((char *)net->hidden_weights);
    free((char *)net->hidden_prev_weights);

    free((char *)net);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

BPNN *bpnn_create(n_in, n_hidden, n_out) int n_in, n_hidden, n_out;
{

    BPNN *newnet;

    newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
    bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
    bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
    bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
    bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
    bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
    bpnn_randomize_row(newnet->target, n_out);
    return (newnet);
}


void bpnn_layerforward(l1, l2, conn, n1, n2) float *l1, *l2, *conn;
int n1, n2;
{
    float sum;
    int j, k;

    /*** Set up thresholding unit ***/
    l1[0] = 1.0;
#ifdef OMP_OFFLOAD
#pragma omp target enter data map(always, to: l1[:n1+1], l2[:n2+1], conn[:(n1+1)*(n2+1)])
//    for (j = 0; j <= n1; j++) {
//#pragma omp target enter data map(always, to: conn[j][:n2+1])
//    }
#pragma omp target teams distribute private(k,j)
    // sum no need to reduction
#else
    omp_set_num_threads(NUM_THREAD);
#pragma omp parallel for shared(conn, n1, n2, l1) private(k, j) reduction(     \
    + : sum) schedule(static)
#endif
    /*** For each unit in second layer ***/
    for (j = 1; j <= n2; j++) {

        /*** Compute weighted sum of its inputs ***/
        float sum = 0.0;
        for (k = 0; k <= n1; k++) {
            sum += conn[k*(n2+1)+j] * l1[k];
        }
        l2[j] = squash(sum);
    }
#ifdef OMP_OFFLOAD
#pragma omp target exit data map(always, from: l2[:n2+1])
#endif
}

// extern "C"
void bpnn_output_error(delta, target, output, nj, err) float *delta, *target,
    *output, *err;
int nj;
{
    int j;
    float o, t, errsum;
    errsum = 0.0;
    for (j = 1; j <= nj; j++) {
        o = output[j];
        t = target[j];
        delta[j] = o * (1.0 - o) * (t - o);
        errsum += ABS(delta[j]);
    }
    *err = errsum;
}


void bpnn_hidden_error(delta_h, nh, delta_o, no, who, hidden,
                       err) float *delta_h,
    *delta_o, *hidden, *who, *err;
int nh, no;
{
    int j, k;
    float h, sum, errsum;

    errsum = 0.0;
    for (j = 1; j <= nh; j++) {
        h = hidden[j];
        sum = 0.0;
        for (k = 1; k <= no; k++) {
            sum += delta_o[k] * who[j*(no+1)+k];
        }
        delta_h[j] = h * (1.0 - h) * sum;
        errsum += ABS(delta_h[j]);
    }
    *err = errsum;
}


void bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw) float *delta, *ly,
    *w, *oldw;
int ndelta, nly;
{
    float new_dw;
    int k, j;
    ly[0] = 1.0;
    // eta = 0.3;
    // momentum = 0.3;

#ifdef OMP_OFFLOAD
#pragma omp target enter data map(always, to: oldw[:(nly+1)*(ndelta+1)], w[:(nly+1)*(ndelta+1)], delta[:ndelta+1], ly[:nly+1])
//    for (int k = 0; k <= nly; k++) {
//#pragma omp target enter data map(always, to: oldw[k][:ndelta+1], w[k][:ndelta+1])
//    }
#pragma omp target teams distribute private(j, k, new_dw), firstprivate(ndelta, nly)
#else
    omp_set_num_threads(NUM_THREAD);
#pragma omp parallel for shared(oldw, w, delta) private(                       \
    j, k, new_dw) firstprivate(ndelta, nly)
#endif
    for (j = 1; j <= ndelta; j++) {
        for (k = 0; k <= nly; k++) {
            new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k*(ndelta+1)+j]));
            w[k*(ndelta+1)+j] += new_dw;
            oldw[k*(ndelta+1)+j] = new_dw;
        }
    }

#ifdef OMP_OFFLOAD
//#pragma omp target exit data map(always, from: delta[:ndelta+1], ly[:nly+1])
#pragma omp target exit data map(always, from: oldw[:(nly+1)*(ndelta+1)], w[:(nly+1)*(ndelta+1)], delta[:ndelta+1], ly[:nly+1])
//    for (int k = 0; k <= nly; k++) {
//#pragma omp target exit data map(always, from: oldw[k][:ndelta+1], w[k][:ndelta+1])
//    }
#endif
}


void bpnn_feedforward(net) BPNN *net;
{
    int in, hid, out;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    /*** Feed forward input activations. ***/
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights,
                      in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                      hid, out);
}


void bpnn_train(net, eo, eh) BPNN *net;
float *eo, *eh;
{
    int in, hid, out;
    float out_err, hid_err;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    /*** Feed forward input activations. ***/
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights,
                      in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                      hid, out);

    /*** Compute error on output and hidden units. ***/
    bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                      &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                      net->hidden_weights, net->hidden_units, &hid_err);
    *eo = out_err;
    *eh = hid_err;

    /*** Adjust input and hidden weights. ***/
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                        net->hidden_weights, net->hidden_prev_weights);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                        net->input_weights, net->input_prev_weights);
}


void bpnn_save(net, filename) BPNN *net;
char *filename;
{
    int n1, n2, n3, i, j, memcnt;
    float dvalue, *w;
    char *mem;
    FILE *pFile;
    pFile = fopen(filename, "w+");

    n1 = net->input_n;
    n2 = net->hidden_n;
    n3 = net->output_n;
    printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);

    fwrite((int *)&n1, sizeof(int), 1, pFile);
    fwrite((int *)&n2, sizeof(int), 1, pFile);
    fwrite((int *)&n3, sizeof(int), 1, pFile);

    memcnt = 0;
    w = net->input_weights;
    mem = (char *)malloc((unsigned)((n1 + 1) * (n2 + 1) * sizeof(float)));
    for (i = 0; i <= n1; i++) {
        for (j = 0; j <= n2; j++) {
            dvalue = w[i*(n2+1)+j];
            fastcopy(&mem[memcnt], &dvalue, sizeof(float));
            memcnt += sizeof(float);
        }
    }
    fwrite(mem, (unsigned)(sizeof(float)), (unsigned)((n1 + 1) * (n2 + 1)),
           pFile);
    free(mem);

    memcnt = 0;
    w = net->hidden_weights;
    mem = (char *)malloc((unsigned)((n2 + 1) * (n3 + 1) * sizeof(float)));
    for (i = 0; i <= n2; i++) {
        for (j = 0; j <= n3; j++) {
            dvalue = w[i*(n3+1)+j];
            fastcopy(&mem[memcnt], &dvalue, sizeof(float));
            memcnt += sizeof(float);
        }
    }
    fwrite(mem, sizeof(float), (unsigned)((n2 + 1) * (n3 + 1)), pFile);
    free(mem);

    fclose(pFile);
    return;
}


BPNN *bpnn_read(filename) char *filename;
{
    char *mem;
    BPNN *new;
    int fd, n1, n2, n3, i, j, memcnt;

    if ((fd = open(filename, 0, 0644)) == -1) {
        return (NULL);
    }

    printf("Reading '%s'\n", filename);

    read(fd, (int *)&n1, sizeof(int));
    read(fd, (int *)&n2, sizeof(int));
    read(fd, (int *)&n3, sizeof(int));
    new = bpnn_internal_create(n1, n2, n3);

    printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
    printf("Reading input weights...");

    memcnt = 0;
    mem = (char *)malloc((unsigned)((n1 + 1) * (n2 + 1) * sizeof(float)));
    read(fd, mem, (n1 + 1) * (n2 + 1) * sizeof(float));
    for (i = 0; i <= n1; i++) {
        for (j = 0; j <= n2; j++) {
            fastcopy(&(new->input_weights[i*(n2+1)+j]), &mem[memcnt], sizeof(float));
            memcnt += sizeof(float);
        }
    }
    free(mem);

    printf("Done\nReading hidden weights...");

    memcnt = 0;
    mem = (char *)malloc((unsigned)((n2 + 1) * (n3 + 1) * sizeof(float)));
    read(fd, mem, (n2 + 1) * (n3 + 1) * sizeof(float));
    for (i = 0; i <= n2; i++) {
        for (j = 0; j <= n3; j++) {
            fastcopy(&(new->hidden_weights[i*(n2+1)+j]), &mem[memcnt], sizeof(float));
            memcnt += sizeof(float);
        }
    }
    free(mem);
    close(fd);

    printf("Done\n");

    bpnn_zero_weights(new->input_prev_weights, n1, n2);
    bpnn_zero_weights(new->hidden_prev_weights, n2, n3);

    return (new);
}
