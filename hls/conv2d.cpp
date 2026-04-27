#include "conv2d.h"

void conv2d(
    fixed_t *input,
    fixed_t *output,
    fixed_t weights[F][1][K][K],
    fixed_t bias[F]
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=INPUT_MEM
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=OUTPUT_MEM
    #pragma HLS INTERFACE s_axilite port=weights bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=bias bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    FILTER_LOOP: for (int f = 0; f < F; f++) {

        fixed_t line_buf[3][IN_W];
        #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

        for (int c = 0; c < IN_W; c++) {
            #pragma HLS PIPELINE II=1
            line_buf[0][c] = input[c];
            line_buf[1][c] = input[IN_W + c];
        }

        int buf_row = 2;
        ROW_LOOP: for (int r = 0; r < OUT_H; r++) {

            int next_row = r + 2;
            for (int c = 0; c < IN_W; c++) {
                #pragma HLS PIPELINE II=1
                line_buf[buf_row][c] = input[next_row * IN_W + c];
            }

            COL_LOOP: for (int c = 0; c < OUT_W; c++) {
                #pragma HLS PIPELINE II=1

                fixed_t sum = bias[f];
                fixed_t products[K][K];
                #pragma HLS ARRAY_PARTITION variable=products complete dim=0

                KR_LOOP: for (int kr = 0; kr < K; kr++) {
                    #pragma HLS UNROLL
                    KC_LOOP: for (int kc = 0; kc < K; kc++) {
                        #pragma HLS UNROLL
                        int row_idx = (r + kr) % 3;
                        products[kr][kc] = line_buf[row_idx][c + kc] * weights[f][0][kr][kc];
                    }
                }

                for (int kr = 0; kr < K; kr++) {
                    #pragma HLS UNROLL
                    for (int kc = 0; kc < K; kc++) {
                        #pragma HLS UNROLL
                        sum += products[kr][kc];
                    }
                }

                int out_idx = f * OUT_H * OUT_W + r * OUT_W + c;
                output[out_idx] = sum;
            }

            buf_row = (buf_row == 2) ? 0 : buf_row + 1;
        }
    }
}