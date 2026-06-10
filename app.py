import numpy as np
import tensorflow as tf
import cv2
import gradio as gr

CLASS_NAMES = ['helmet', 'no_helmet']

# ---------------------------------------------------------------
# Load the two trained models
# ---------------------------------------------------------------
cnn_model = tf.keras.models.load_model('custom_cnn.keras')
eff_model = tf.keras.models.load_model('efficientnetv2.keras')

# ---------------------------------------------------------------
# Grad-CAM helpers
# ---------------------------------------------------------------
def get_backbone(model):
    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            return l
    return None

def last_conv_name(src):
    for l in reversed(src.layers):
        try:
            if len(l.output.shape) == 4:
                return l.name
        except Exception:
            continue
    return None

def build_grad_model(model):
    backbone = get_backbone(model)
    src = backbone if backbone is not None else model
    name = last_conv_name(src)
    return tf.keras.models.Model(model.inputs, [src.get_layer(name).output, model.outputs[0]])

gm_cnn = build_grad_model(cnn_model)
gm_eff = build_grad_model(eff_model)

def compute_gradcam(grad_model, x):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        cls = tf.argmax(preds[0])
        loss = preds[:, cls]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.squeeze(conv_out[0] @ pooled[..., None])
    hm = tf.maximum(hm, 0) / (tf.reduce_max(hm) + 1e-8)
    return hm.numpy()

def overlay_heatmap(img224, hm):
    hm = cv2.resize(hm, (224, 224))
    hm = np.uint8(255 * hm)
    hm = cv2.GaussianBlur(hm, (13, 13), 0)
    hm = cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8(0.6 * img224 + 0.4 * hm)

# ---------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------
def predict(image):
    if image is None:
        return None, None, "Please upload an image."

    img224 = cv2.resize(image, (224, 224)).astype('float32')
    x_norm = np.expand_dims(img224 / 255.0, 0)   # Custom CNN: [0,1]
    x_raw = np.expand_dims(img224, 0)            # EfficientNetV2: [0,255]

    p_cnn = cnn_model.predict(x_norm, verbose=0)[0]
    p_eff = eff_model.predict(x_raw, verbose=0)[0]

    cnn_cls = int(np.argmax(p_cnn)); cnn_conf = float(np.max(p_cnn))
    eff_cls = int(np.argmax(p_eff)); eff_conf = float(np.max(p_eff))

    hm_cnn = compute_gradcam(gm_cnn, x_norm)
    hm_eff = compute_gradcam(gm_eff, x_raw)

    img8 = img224.astype('uint8')
    cam_cnn = overlay_heatmap(img8, hm_cnn)
    cam_eff = overlay_heatmap(img8, hm_eff)

    summary = (
        f"Custom CNN  ->  {CLASS_NAMES[cnn_cls]}  ({cnn_conf*100:.1f}%)\n"
        f"EfficientNetV2  ->  {CLASS_NAMES[eff_cls]}  ({eff_conf*100:.1f}%)"
    )
    if cnn_cls != eff_cls:
        summary += "\n\n(The two models disagree on this image.)"

    return cam_cnn, cam_eff, summary

# ---------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------
with gr.Blocks(title="Explainable Helmet-Use Classification") as demo:
    gr.Markdown(
        "# Explainable Helmet-Use Classification\n"
        "Upload a cropped image of a person's head / rider. "
        "Two models (a custom CNN and an EfficientNetV2 transfer-learning model) "
        "predict whether a helmet is worn, and Grad-CAM shows where each model looks."
    )
    with gr.Row():
        inp = gr.Image(type="numpy", label="Input image")
        out_text = gr.Textbox(label="Predictions", lines=4)
    with gr.Row():
        out_cnn = gr.Image(label="Custom CNN — Grad-CAM")
        out_eff = gr.Image(label="EfficientNetV2 — Grad-CAM")
    btn = gr.Button("Classify", variant="primary")
    btn.click(fn=predict, inputs=inp, outputs=[out_cnn, out_eff, out_text])

    gr.Markdown(
        "Classes: **helmet** / **no_helmet**. "
        "Models trained on bounding-box crops from a YOLO helmet dataset."
    )

demo.launch()
