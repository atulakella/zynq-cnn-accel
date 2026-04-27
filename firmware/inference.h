#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

// ================================================================
//  SW Inference Layers
//  Runs on ARM Cortex-A9 (PS side) after HW accelerator finishes.
//
//  Pipeline:
//    output_buf[8*26*26]  (int16, from HW via AXI-Master → DDR)
//      → relu_int16()         in-place, int16 domain
//      → maxpool2x2_to_float()  2×2 stride-2, int16→float
//      → pool_output_f[8*13*13]
//      → fc_layer()           1352 → 10, random weights for now
//      → softmax()            numerically stable (max-subtract)
//      → argmax()             predicted class 0–9
//
//  All large buffers (fc_weights 54 KB, etc.) are static inside
//  inference.c — never on stack. Zynq bare-metal stack is ~8 KB.
// ================================================================

// Call once at startup before first inference.
// Fills fc_weights with reproducible LCG random floats (±0.05).
// Replace with trained weights when available.
void inference_init(void);
void relu_int16(int16_t *data, int len);
void maxpool2x2_to_float(int16_t *in, float *out);
void fc_layer(float *in, float *out);
// Full SW pipeline. Reads from hw_output[8*26*26], writes result to
// predicted_class. Returns argmax of softmax output.
// Also fills softmax_scores[10] for caller to inspect/print.
int inference_run(int16_t *hw_output, float *softmax_scores_out);

// Utility: print top-3 class scores to UART.
// Uses printf (not xil_printf) — %f works here.
void inference_print_top3(float *softmax_scores);

#endif // INFERENCE_H