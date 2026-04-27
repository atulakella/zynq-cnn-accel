# MNIST CNN Accelerator on Zynq-7000

A hardware CNN accelerator built on the Zybo Z7-10. Conv2D IP runs on the FPGA fabric as a custom HLS IP block. ReLU, MaxPool, FC, and Softmax run on the ARM Cortex-A9 in software.

Full writeup: [Medium post]()

---

## Results

| Stage         | Time     |
|---------------|----------|
| HW Conv2D     | 0.344 ms |
| SW layers     | 0.360 ms |
| E2E total     | 0.704 ms |
| Conv speedup  | 2.67×    |

**Predicted:** 7 
**Top-3:**  
- [7] = 0.9996  
- [3] = 0.0004  
- [9] = 0.0000  

---

## Repo Structure

| Folder | Contents |
|--------|----------|
| `firmware/` | Bare-metal C main loop, inference layers, MNIST image |
| `hls/` | Conv2D HLS kernel and C simulation testbench |
| `python/` | Trains the Keras model, exports weights to C arrays |
| `results/` | Raw benchmark output from hardware |
| `vivado/` | Block design TCL script and wiring diagram |

---

## Requirements

- Vitis HLS 2024.2  
- Vivado 2024.2  
- Zybo Z7-10 (xc7z010clg400-1) 
- Python 3.x  
- TensorFlow 2.x  
- NumPy  

---

## Notes

Run the Python script first. It trains the model, transposes conv  
weights from NHWC to NCHW, quantizes to Q8 int16, and prints C  
arrays ready to paste into the firmware files.

If UART is silent after flashing, check the MIO pin configuration.  
Details in the blog post.

`inference.c` contains the trained FC weights as a hardcoded float  
array, scroll past the weight values to get to the actual logic.