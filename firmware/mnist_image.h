#ifndef MNIST_IMAGE_H
#define MNIST_IMAGE_H

#include <stdint.h>

// ================================================================
//  MNIST IMAGE — replace array contents with real MNIST digit.
//
//  To generate from real MNIST binary:
//
//    import struct
//    with open('train-images-idx3-ubyte', 'rb') as f:
//        f.read(16)
//        img = [struct.unpack('B', f.read(1))[0] for _ in range(784)]
//    with open('train-labels-idx1-ubyte', 'rb') as f:
//        f.read(8)
//        label = struct.unpack('B', f.read(1))[0]
//    normalized = [int(p / 255.0 * 256) for p in img]
//    print(f"int true_label = {label};")
//    print("int16_t mnist_image[784] = {" + ",".join(str(v) for v in normalized) + "};")
//
//  Normalization: pixel/255.0 * 256 → int16_t range [0, 256].
//  Confirm this matches the normalization used in your HLS C sim.
//
//  Currently: synthetic "ring" pattern (bright border, dark interior).
//  true_label = -1 means synthetic — no ground truth available.
// ================================================================

extern int       true_label;
extern int16_t   mnist_image[784];

#endif // MNIST_IMAGE_H