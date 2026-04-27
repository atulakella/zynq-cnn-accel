import numpy as np

# ── 1. Load MNIST ─────────────────────────────────────────────
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32)  / 255.0

# ── 2. Build model — matches your HW/SW pipeline exactly ──────
# Input: 28x28 → Conv2d(8 filters, 3x3) → 8x26x26
# → ReLU → MaxPool2x2 → 8x13x13 = 1352 → FC(10) → Softmax
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, 3, padding='valid', activation='relu',
                           use_bias=True),   # → 26x26x8
    tf.keras.layers.MaxPooling2D(2),          # → 13x13x8
    tf.keras.layers.Flatten(),               # → 1352
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

# ── 3. Train ───────────────────────────────────────────────────
x_train_r = x_train.reshape(-1, 28, 28, 1)
x_test_r  = x_test.reshape(-1,  28, 28, 1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_r, y_train, epochs=5, batch_size=128,
          validation_data=(x_test_r, y_test))

# ── 4. Evaluate ────────────────────────────────────────────────
loss, acc = model.evaluate(x_test_r, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.2f}%")

# ── 5. Extract FC weights ──────────────────────────────────────
# FC layer: kernel shape = (1352, 10), bias shape = (10,)
fc_layer  = model.layers[-1]
fc_w      = fc_layer.get_weights()[0]   # (1352, 10)
fc_b      = fc_layer.get_weights()[1]   # (10,)

print(f"FC weight shape: {fc_w.shape}")
print(f"FC bias shape:   {fc_b.shape}")

# ── 6. Extract conv weights for reference ──────────────────────
# You'll need these to load into HLS weights_buf too
conv_layer = model.layers[0]
conv_w     = conv_layer.get_weights()[0]  # (3, 3, 1, 8)
conv_b     = conv_layer.get_weights()[1]  # (8,)

# Reorder from Keras (3,3,1,8) → HLS (8,1,3,3)
conv_w_reordered = conv_w.transpose(3, 2, 0, 1)
conv_w_q8 = np.clip(np.round(conv_w_reordered * 256), -32768, 32767).astype(np.int16)
conv_b_q8 = np.clip(np.round(conv_b * 256), -32768, 32767).astype(np.int16)
# ── 7. Generate C output ───────────────────────────────────────
def fmt_float_array_1d(name, arr):
    vals = ", ".join(f"{v:.8f}f" for v in arr.flatten())
    return f"static const float {name}[] = {{{vals}}};"

def fmt_int16_array(name, arr):
    vals = ", ".join(str(v) for v in arr.flatten())
    return f"static const int16_t {name}[] = {{{vals}}};"

print("\n\n// ── PASTE INTO inference.c ───────────────────────────")
print(fmt_float_array_1d("trained_fc_weights_flat", fc_w))   # 1352*10 = 13520 values
print(fmt_float_array_1d("trained_fc_bias", fc_b))           # 10 values

print("\n\n// ── PASTE INTO main.c (replace weights_buf init) ────")
print(fmt_int16_array("trained_conv_weights_flat", conv_w_q8))  # 8*3*3 = 72 values
print(fmt_int16_array("trained_conv_bias", conv_b_q8))          # 8 values

# ── 8. Also generate the test image ───────────────────────────
# Take first test image
img   = x_test[0]
label = y_test[0]
img_q = np.clip(np.round(img * 256), 0, 32767).astype(np.int16)
vals  = ", ".join(str(v) for v in img_q.flatten())
print(f"\n\n// ── PASTE INTO mninst_image.c ───────────────────────")
print(f"int true_label = {label};")
print(f"int16_t mnist_image[784] = {{{vals}}};")