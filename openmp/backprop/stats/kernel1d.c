
    for (j = 1; j <= ndelta; j++) {
        for (k = 0; k <= nly; k++) {
            new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k*(ndelta+1)+j]));
            w[k*(ndelta+1)+j] += new_dw;
            oldw[k*(ndelta+1)+j] = new_dw;
        }
    }
    for (j = 1; j <= n2; j++) {

        /*** Compute weighted sum of its inputs ***/
        float sum = 0.0;
        for (k = 0; k <= n1; k++) {
            sum += conn[k*(n2+1)+j] * l1[k];
        }
        l2[j] = squash(sum);
    }
