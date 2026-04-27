#include <stdio.h>
#include <math.h>
#include "conv2d.h"

int main() {
    static fixed_t input[IN_H * IN_W];
    static fixed_t output[F * OUT_H * OUT_W];
    static fixed_t weights[F][1][K][K];
    static fixed_t bias[F];

    for (int i = 0; i < IN_H * IN_W; i++) input[i] = 1.0f;
    for (int f = 0; f < F; f++) {
        bias[f] = 0.0f;
        for (int kr = 0; kr < K; kr++)
            for (int kc = 0; kc < K; kc++)
                weights[f][0][kr][kc] = 1.0f;
    }

    conv2d(input, output, weights, bias);

    int pass = 1;
    for (int i = 0; i < F * OUT_H * OUT_W; i++) {
        if (fabsf((float)output[i] - 9.0f) > 0.1f) {
            printf("FAIL at index %d: got %f expected 9.0\n", i, (float)output[i]);
            pass = 0;
            break;
        }
    }

    if (pass) printf("PASS: all %d output elements = 9.0\n", F * OUT_H * OUT_W);
    return pass ? 0 : 1;
}