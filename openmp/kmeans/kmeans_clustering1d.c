/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**
 * **/
/**   Edited by: Sang-Ha  Lee
 * **/
/**				 University of Virginia
 * **/
/**
 * **/
/**   Description:	No longer supports fuzzy c-means clustering;
 * **/
/**					only regular k-means clustering.
 * **/
/**					Simplified for main functionality: regular
 * k-means	**/
/**					clustering.
 * **/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#include "kmeans1d.h"

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);

int find_nearest_point(float *pt,                  /* [nfeatures] */
                       int nfeatures, float *pts, /* [npts*nfeatures] */
                       int npts) {
    int index, i;
    float min_dist = FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i = 0; i < npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts+i*nfeatures, nfeatures); /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return (index);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline float euclid_dist_2(float *pt1, float *pt2, int numdims) {
    int i;
    float ans = 0.0;

    for (i = 0; i < numdims; i++)
        ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

    return (ans);
}

/*----< kmeans_clustering() >---------------------------------------------*/
float *kmeans_clustering1d(float *feature, /* in: [npoints*nfeatures] */
                          int nfeatures, int npoints, int nclusters,
                          float threshold, int *membership) /* out: [npoints] */
{

    int i, j, k, n = 0, index, loop = 0;
    int *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float *new_centers;  /* [nclusters*nfeatures] */
    float *clusters;     /* out: [nclusters*nfeatures] */
    float delta;

    int nthreads;
    int *partial_new_centers_len;
    float *partial_new_centers;

    nthreads = omp_get_max_threads();

    /* allocate space for returning variable clusters[] */
    //clusters = (float **)malloc(nclusters * sizeof(float *));
    clusters = (float *)malloc(nclusters * nfeatures * sizeof(float));
    //for (i = 1; i < nclusters; i++)
    //    clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        // n = (int)rand() % npoints;
        for (j = 0; j < nfeatures; j++)
        clusters[i*nfeatures+j] = feature[n*nfeatures+j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int *)calloc(nclusters, sizeof(int));

    //new_centers = (float **)malloc(nclusters * sizeof(float *));
    new_centers = (float *)calloc(nclusters * nfeatures, sizeof(float));
    //for (i = 1; i < nclusters; i++)
    //    new_centers[i] = new_centers[i - 1] + nfeatures;


    //partial_new_centers_len = (int **)malloc(nthreads * sizeof(int *));
    partial_new_centers_len =
        (int *)calloc(nthreads * nclusters, sizeof(int));
    //for (i = 1; i < nthreads; i++)
    //    partial_new_centers_len[i] = partial_new_centers_len[i - 1] + nclusters;

    partial_new_centers = (float *)calloc(nthreads * nclusters * nfeatures, sizeof(float ));
    /*
    partial_new_centers[0] =
        (float **)malloc(nthreads * nclusters * sizeof(float *));
    for (i = 1; i < nthreads; i++)
        partial_new_centers[i] = partial_new_centers[i - 1] + nclusters;

    for (i = 0; i < nthreads; i++) {
        for (j = 0; j < nclusters; j++)
            partial_new_centers[i][j] =
                (float *)calloc(nfeatures, sizeof(float));
    }
    */

    double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
#ifdef OMP_OFFLOAD
#pragma omp target enter data map(to:feature[:npoints*nfeatures], membership[:npoints])
#endif
    do {
        delta = 0.0;
#if defined OMP_OFFLOAD
#pragma omp target enter data map(to: clusters[:nclusters*nfeatures], partial_new_centers[:nthreads*nclusters*nfeatures], partial_new_centers_len[:nclusters*nfeatures])
        {
            int tid = 0;
//#pragma omp target teams distribute parallel for private(i, j, index) firstprivate(npoints, nclusters, nfeatures) reduction(+ : delta)
#pragma omp target teams distribute parallel for private(i, j, index) firstprivate(npoints, nclusters, nfeatures) reduction(+: delta)
#else
#pragma omp parallel shared(feature, clusters, membership,                     \
                            partial_new_centers, partial_new_centers_len)
        {
            int tid = omp_get_thread_num();
#pragma omp for private(i, j, index) firstprivate(                             \
    npoints, nclusters, nfeatures) schedule(static) reduction(+ : delta)
#endif
            for (i = 0; i < npoints; i++) {
                /* find the index of nestest cluster centers */
                index = find_nearest_point(feature+i*nfeatures, nfeatures, clusters,
                                           nclusters);
                /* if membership changes, increase delta by 1 */
                if (membership[i] != index)
//#pragma omp atomic
                    delta += 1.0;

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of all objects located
                       within */
#pragma omp atomic
                partial_new_centers_len[tid*nclusters+index]++;
                for (j = 0; j < nfeatures; j++) {
#pragma omp atomic
                    partial_new_centers[(tid*nclusters+index)*nfeatures+j] += feature[i*nfeatures+j];
                }
            }
        }
#ifdef OMP_OFFLOAD
#pragma omp target exit data map(from: clusters[:nclusters*nfeatures], partial_new_centers[:nthreads*nclusters*nfeatures], partial_new_centers_len[:nclusters*nfeatures])
#endif


        /* let the main thread perform the array reduction */
        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nthreads; j++) {
                new_centers_len[i] += partial_new_centers_len[j*nclusters+i];
                partial_new_centers_len[j*nclusters+i] = 0.0;
                for (k = 0; k < nfeatures; k++) {
                    new_centers[i*nfeatures+k] += partial_new_centers[(j*nclusters+i)*nfeatures+k];
                    partial_new_centers[(j*nclusters+i)*nfeatures+k] = 0.0;
                }
            }
        }
#if 0
        if (loop==0) {
        int a = 0;

        for (i = 0; i < nclusters; i++) {
            //printf("%d ", new_centers_len[i]);
            //a += new_centers_len[i];
            for (int j = 0; j < nfeatures; j++) {
                int index = i*nfeatures + j;
                if (index >= 60 && index <= 70) {
                    printf("%f ", new_centers[index]);
                }
            }
        }
        puts("");
        for (i = 0; i < npoints; i++) {
            //printf("%d ", membership[i]);
        }
        printf("\na: %d npoints: %d delta: %f\n", a, npoints, delta);
        puts("");
       break;
        }
#endif

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nfeatures; j++) {
                if (new_centers_len[i] > 0)
                    clusters[i*nfeatures+j] = new_centers[i*nfeatures+j] / new_centers_len[i];
                new_centers[i*nfeatures+j] = 0.0; /* set back to 0 */
            }
            new_centers_len[i] = 0; /* set back to 0 */
        }

        printf("%d ", loop);
        fflush(stdout);
    } while (delta > threshold && loop++ < 500);
        printf("t0 %lf t1 %lf t2 %lf t3 %lf\n", t0,  t1, t2, t3);
    printf("Loop %d\n", loop);


    free(new_centers);
    free(new_centers_len);

    return clusters;
}
