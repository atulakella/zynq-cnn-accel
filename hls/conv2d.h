#ifndef CONV2D_H
#define CONV2D_H

#include <ap_fixed.h>

#define IN_H    28
#define IN_W    28
#define K       3
#define F       8
#define OUT_H   26
#define OUT_W   26

typedef ap_fixed<16,8> fixed_t;

void conv2d(
    fixed_t *input,
    fixed_t *output,
    fixed_t weights[F][1][K][K],
    fixed_t bias[F]
);

#endif