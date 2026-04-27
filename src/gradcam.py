import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv2d_2'):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(original_pil, heatmap, alpha=0.45, colormap='plasma'):
    cmap = plt.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored).resize(original_pil.size, Image.LANCZOS)
    original_arr = np.array(original_pil.convert('RGB')).astype(float)
    colored_arr = np.array(colored_img).astype(float)
    blended = (colored_arr * alpha + original_arr * (1 - alpha)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def gradcam_for_image(img_array, original_pil, model, alpha=0.45):
    heatmap = make_gradcam_heatmap(img_array, model)
    return overlay_heatmap(original_pil, heatmap, alpha=alpha)
