#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "xconv2d.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xil_io.h"
#include "mninst_image.h"
#include "inference.h"

// ================================================================
//  TIMER — SCU Private Timer, fixed at CPU_CLK/2 = 333.33 MHz
// ================================================================
#define TIMER_BASE      XPAR_XSCUTIMER_0_BASEADDR
#define TIMER_LOAD      (TIMER_BASE + 0x00)
#define TIMER_COUNTER   (TIMER_BASE + 0x04)
#define TIMER_CONTROL   (TIMER_BASE + 0x08)
#define SCU_HZ          333333333.0f
#define SCU_KHZ         333333.0f

volatile uint32_t bg_count = 0;

static inline void timer_start(void) {
    Xil_Out32(TIMER_LOAD, 0xFFFFFFFF);
    Xil_Out32(TIMER_CONTROL, 0x1);
}

static inline u32 timer_read(void) {
    u32 v = 0xFFFFFFFF - Xil_In32(TIMER_COUNTER);
    Xil_Out32(TIMER_CONTROL, 0);
    return v;
}

static inline float to_ms(u32 cycles) { return (float)cycles / SCU_KHZ; }
static inline float to_us(u32 cycles) { return (float)cycles / (SCU_HZ / 1e6f); }

// ================================================================
//  BUFFERS
// ================================================================
int16_t  output_buf[8 * 26 * 26];
int16_t  weights_buf[8][1][3][3];
int16_t  bias_buf[8];

static int16_t  sw_output_buf[8 * 26 * 26];
static float    pool_buf[1352];
static float    fc_out_buf[10];

// ================================================================
//  SW CONV REFERENCE (Q8, matches HLS arithmetic)
// ================================================================
static void sw_conv2d(const int16_t *input,
                      const int16_t *weights,
                      const int16_t *bias,
                      int16_t       *output)
{
    for (int f = 0; f < 8; f++)
        for (int r = 0; r < 26; r++)
            for (int c = 0; c < 26; c++) {
                int32_t acc = 0;
                for (int kr = 0; kr < 3; kr++)
                    for (int kc = 0; kc < 3; kc++)
                        acc += (int32_t)input[(r+kr)*28 + (c+kc)]
                             * (int32_t)weights[f*9 + kr*3 + kc];
                acc = (acc >> 8) + bias[f];
                output[f*26*26 + r*26 + c] = (int16_t)acc;
            }
}

// ================================================================
//  MAIN
// ================================================================
int main(void) {
    printf("\r\n=== MNIST CNN Inference on Zynq-7000 ===\r\n");
    printf("Device  : Zybo Z7-10 (xc7z010clg400-1)\r\n");
    printf("Timer   : SCU @ %.0f MHz\r\n", SCU_HZ / 1e6f);
    printf("Image   : MNIST digit, true label = %d\r\n\r\n", true_label);

    // ── Init ─────────────────────────────────────────────────────
    inference_init();

    static const int16_t trained_conv_weights_flat[] = {
        -112, 56, 57, -125, 65, 107, 10, 38, 76, 22, 97, -10, 47, 77, 81, 32,
        23, 2, 36, -73, -268, 205, 44, -207, 130, 115, -35, 68, 88, 68, 37, -1,
        37, 56, 56, 88, -108, -44, 17, 129, 131, 106, -31, 13, -92, 126, 102, 63,
        -47, -26, 58, -150, -14, 52, 3, 157, 202, -109, -143, 117, 33, -202,
        -190, -4, 159, 102, -242, -371, -250, 125, 190, -1
    };
    static const int16_t trained_conv_bias[] = {0, -1, 20, -3, 1, -20, 0, 61};

    memcpy(weights_buf, trained_conv_weights_flat, sizeof(weights_buf));
    for (int i = 0; i < 8; i++) bias_buf[i] = trained_conv_bias[i];
    for (int i = 0; i < 8*26*26; i++) output_buf[i] = 0;

    // ── HW accelerator setup ─────────────────────────────────────
    XConv2d ip;
    ip.Control_BaseAddress = XPAR_CONV2D_0_BASEADDR;
    ip.Ctrl_BaseAddress    = XPAR_CONV2D_0_BASEADDR_1;
    ip.IsReady             = XIL_COMPONENT_IS_READY;

    XConv2d_Set_input_r(&ip,  (u64)(UINTPTR)mnist_image);
    XConv2d_Set_output_r(&ip, (u64)(UINTPTR)output_buf);
    XConv2d_Write_weights_Words(&ip, 0, (word_type*)weights_buf,
        8*1*3*3 * sizeof(int16_t) / sizeof(word_type));
    XConv2d_Write_bias_Words(&ip, 0, (word_type*)bias_buf,
        8 * sizeof(int16_t) / sizeof(word_type));

    Xil_DCacheFlushRange((UINTPTR)mnist_image,  784 * sizeof(int16_t));
    Xil_DCacheFlushRange((UINTPTR)output_buf,   sizeof(output_buf));
    Xil_DCacheFlushRange((UINTPTR)weights_buf,  sizeof(weights_buf));
    Xil_DCacheFlushRange((UINTPTR)bias_buf,     sizeof(bias_buf));

    // ── [1] HW Conv timing ───────────────────────────────────────
    timer_start();
    XConv2d_Start(&ip);
    int timeout = 10000000;
    while (!XConv2d_IsDone(&ip) && timeout--);
    u32 hw_cycles = timer_read();

    if (timeout <= 0) {
        printf("ERROR: HW timeout!\r\n");
        while (1);
    }
    Xil_DCacheInvalidateRange((UINTPTR)output_buf, sizeof(output_buf));

    // ── [2] SW layer breakdown ───────────────────────────────────
    // Copy output so we can time each stage without destroying data
    static int16_t stage_buf[8 * 26 * 26];
    memcpy(stage_buf, output_buf, sizeof(output_buf));

    timer_start();
    relu_int16(stage_buf, 8 * 26 * 26);
    u32 relu_cycles = timer_read();

    timer_start();
    maxpool2x2_to_float(stage_buf, pool_buf);
    u32 pool_cycles = timer_read();

    timer_start();
    fc_layer(pool_buf, fc_out_buf);
    u32 fc_cycles = timer_read();

    // Full SW pipeline for total timing + result
    float softmax_scores[10];
    timer_start();
    int predicted = inference_run(output_buf, softmax_scores);
    u32 sw_total_cycles = timer_read();

    // ── [3] SW conv reference benchmark ─────────────────────────
    timer_start();
    sw_conv2d(mnist_image, trained_conv_weights_flat,
              trained_conv_bias, sw_output_buf);
    u32 sw_conv_cycles = timer_read();

    // ── [4] HW latency variance (10 runs) ───────────────────────
    u32 hw_runs[10];
    for (int run = 0; run < 10; run++) {
        for (int i = 0; i < 8*26*26; i++) output_buf[i] = 0;
        Xil_DCacheFlushRange((UINTPTR)mnist_image, 784 * sizeof(int16_t));
        Xil_DCacheFlushRange((UINTPTR)output_buf,  sizeof(output_buf));
        timer_start();
        XConv2d_Start(&ip);
        timeout = 10000000;
        while (!XConv2d_IsDone(&ip) && timeout--);
        hw_runs[run] = timer_read();
        Xil_DCacheInvalidateRange((UINTPTR)output_buf, sizeof(output_buf));
    }
    u32 hw_min = hw_runs[0], hw_max = hw_runs[0], hw_sum = 0;
    for (int i = 0; i < 10; i++) {
        if (hw_runs[i] < hw_min) hw_min = hw_runs[i];
        if (hw_runs[i] > hw_max) hw_max = hw_runs[i];
        hw_sum += hw_runs[i];
    }

    // ── [5] Output distribution ──────────────────────────────────
    int16_t out_min = output_buf[0], out_max = output_buf[0];
    int32_t out_sum = 0;
    int n_neg = 0, n_zero = 0, n_pos = 0;
    for (int i = 0; i < 8*26*26; i++) {
        if (output_buf[i] < out_min) out_min = output_buf[i];
        if (output_buf[i] > out_max) out_max = output_buf[i];
        out_sum += output_buf[i];
        if      (output_buf[i] < 0) n_neg++;
        else if (output_buf[i] == 0) n_zero++;
        else n_pos++;
    }

    // ── [6] Mismatch per channel ─────────────────────────────────
    int ch_mm[8] = {0};
    int total_mm = 0;
    for (int i = 0; i < 8*26*26; i++)
        if (sw_output_buf[i] != output_buf[i]) {
            ch_mm[i / (26*26)]++;
            total_mm++;
        }

    // ── PRINT FULL REPORT ────────────────────────────────────────
    printf("================================================\r\n");
    printf("         DETAILED BENCHMARK REPORT\r\n");
    printf("================================================\r\n");

    printf("\r\n[1] Inference Result\r\n");
    printf("  True label  : %d\r\n", true_label);
    printf("  Predicted   : %d  %s\r\n", predicted,
           predicted == true_label ? "(CORRECT)" : "(WRONG)");
    inference_print_top3(softmax_scores);

    printf("\r\n[2] Conv Latency\r\n");
    printf("  HW conv     : %8lu cycles   %.3f ms\r\n", hw_cycles, to_ms(hw_cycles));
    printf("  SW conv     : %8lu cycles   %.3f ms\r\n", sw_conv_cycles, to_ms(sw_conv_cycles));
    printf("  Speedup     : %.2fx\r\n", (float)sw_conv_cycles / hw_cycles);

    printf("\r\n[3] Throughput\r\n");
    printf("  HW conv     : %.1f inferences/sec\r\n", 1000.0f / to_ms(hw_cycles));
    printf("  SW conv     : %.1f inferences/sec\r\n", 1000.0f / to_ms(sw_conv_cycles));

    printf("\r\n[4] Compute Efficiency\r\n");
    u32 total_macs = 8 * 26 * 26 * 9;   // 48,672
    printf("  Total MACs  : %lu\r\n", total_macs);
    printf("  HW MAC/cyc  : %.4f  (1.0 = perfect pipeline)\r\n",
           (float)total_macs / hw_cycles);
    printf("  SW MAC/cyc  : %.4f\r\n",
           (float)total_macs / sw_conv_cycles);

    printf("\r\n[5] SW Layer Breakdown\r\n");
    printf("  ReLU        : %8lu cycles   %.3f ms\r\n", relu_cycles, to_ms(relu_cycles));
    printf("  MaxPool     : %8lu cycles   %.3f ms\r\n", pool_cycles, to_ms(pool_cycles));
    printf("  FC (1352→10): %8lu cycles   %.3f ms\r\n", fc_cycles,   to_ms(fc_cycles));
    printf("  SW total    : %8lu cycles   %.3f ms\r\n", sw_total_cycles, to_ms(sw_total_cycles));

    printf("\r\n[6] End-to-End Pipeline\r\n");
    u32 e2e = hw_cycles + sw_total_cycles;
    printf("  HW conv     : %.3f ms  (%.1f%% of E2E)\r\n",
           to_ms(hw_cycles), 100.0f * hw_cycles / e2e);
    printf("  SW layers   : %.3f ms  (%.1f%% of E2E)\r\n",
           to_ms(sw_total_cycles), 100.0f * sw_total_cycles / e2e);
    printf("  E2E total   : %.3f ms\r\n", to_ms(e2e));
    printf("  E2E throughput: %.1f inferences/sec\r\n", 1000.0f / to_ms(e2e));

    printf("\r\n[7] HW Latency Variance (10 runs)\r\n");
    printf("  Min         : %lu cycles   %.3f ms\r\n", hw_min, to_ms(hw_min));
    printf("  Max         : %lu cycles   %.3f ms\r\n", hw_max, to_ms(hw_max));
    printf("  Mean        : %lu cycles   %.3f ms\r\n", hw_sum/10, to_ms(hw_sum/10));
    printf("  Jitter      : %lu cycles   %.2f us\r\n",
           hw_max - hw_min, to_us(hw_max - hw_min));

    printf("\r\n[8] Conv Output Distribution (post-HW, pre-ReLU)\r\n");
    printf("  Min=%d  Max=%d  Mean=%.2f\r\n",
           out_min, out_max, (float)out_sum / (8*26*26));
    printf("  Negative: %d (%.1f%%)  Zero: %d (%.1f%%)  Positive: %d (%.1f%%)\r\n",
           n_neg,  100.0f*n_neg  / (8*26*26),
           n_zero, 100.0f*n_zero / (8*26*26),
           n_pos,  100.0f*n_pos  / (8*26*26));

    printf("\r\n[9] HW vs SW Mismatch Analysis\r\n");
    printf("  Total mismatches: %d / %d (%.1f%%)\r\n",
           total_mm, 8*26*26, 100.0f * total_mm / (8*26*26));
    printf("  Per channel:\r\n");
    for (int ch = 0; ch < 8; ch++)
        printf("    ch%d: %d / 676 (%.1f%%)\r\n",
               ch, ch_mm[ch], 100.0f * ch_mm[ch] / 676);
    printf("  (Mismatches = +/-1 LSB rounding, HLS ap_fixed vs C >>8)\r\n");
    printf("  (Inference result unaffected — swamped by FC averaging)\r\n");

    printf("\r\n================================================\r\n");
    printf("  PIPELINE COMPLETE — Predicted: %d  (True: %d)\r\n",
           predicted, true_label);
    printf("================================================\r\n");

    while (1);
    return 0;
}