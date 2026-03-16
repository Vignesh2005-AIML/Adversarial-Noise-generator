import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import warnings

# Optional: reduce tensorflow INFO noise in console (not the yellow Streamlit banner)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hides INFO, shows WARNING+

st.set_page_config(page_title="FGSM Adversarial Demo", layout="wide")

@st.cache_resource(show_spinner=False)
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()
loss_object = tf.keras.losses.CategoricalCrossentropy()

# ---------- helpers ----------
def pil_to_preprocess(pil_img, target_size=(224, 224)):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_resized = pil_img.resize(target_size, Image.BILINEAR)
    arr = np.array(pil_resized, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    pre = preprocess_input(batch)  # maps to [-1,1]
    return pre, arr

def deprocess(pre_batch):
    x = pre_batch.copy()
    x = (x + 1.0) * 127.5
    return np.clip(x, 0, 255).astype(np.uint8)

def create_adversarial_pattern(input_image, input_label_onehot):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label_onehot, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def predict_topk(np_batch, topk=3):
    preds = model.predict(np_batch)
    decoded = decode_predictions(preds, top=topk)[0]
    return decoded, preds

# ---------- UI ----------
st.title("    Adversarial Image Generator ")
st.markdown("Upload an image, set epsilon (strength), and create an untargeted FGSM adversarial example.")

left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("Upload image (jpg, png)", type=["jpg", "jpeg", "png"])
    eps = st.slider("Epsilon (perturbation strength)", min_value=0.0, max_value=0.1,
                    value=0.015, step=0.001,
                    help="Epsilon is in the preprocessed [-1,1] domain. Pixel change ≈ eps * 127.5.")
    run_btn = st.button("Create adversarial")

with right:
    st.markdown("**Model:** MobileNetV2 (pretrained on ImageNet)")
    st.markdown("- Uses the model's *predicted* label as the 'true' label (untargeted).")
    st.markdown("- Increase epsilon to strengthen attack (but it becomes more visible).")
    st.markdown("- Download button appears under the Adversarial image.")

if uploaded is not None:
    try:
        pil_img = Image.open(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not read uploaded image: {e}")
        pil_img = None

    if pil_img is not None:
        st.subheader("Uploaded image (preview)")
        # display the uploaded file preview but limit its displayed height via 'width' param
        # compute a moderate width based on container: e.g., 450 px
        st.image(pil_img, caption="Uploaded (full file)", use_container_width=False, width=450)

        if run_btn:
            with st.spinner("Generating adversarial example..."):
                # preprocess to 224x224 for the model
                prep, orig_arr = pil_to_preprocess(pil_img, target_size=(224, 224))
                prep_tf = tf.convert_to_tensor(prep, dtype=tf.float32)

                # original predictions
                decoded_orig, pred_orig = predict_topk(prep)
                st.subheader("Original predictions (top-3)")
                for _class, desc, prob in decoded_orig:
                    st.write(f"- {desc} ({prob:.4f})")

                # build one-hot of predicted label (untargeted)
                pred_label = np.argmax(pred_orig[0])
                label_onehot = tf.one_hot(pred_label, 1000)
                label_onehot = tf.reshape(label_onehot, (1, 1000))

                # FGSM pattern and adversarial image
                pattern = create_adversarial_pattern(prep_tf, label_onehot)
                adv = prep_tf + eps * pattern
                adv = tf.clip_by_value(adv, -1.0, 1.0)
                adv_np = adv.numpy()

                # adversarial predictions
                decoded_adv, pred_adv = predict_topk(adv_np)
                st.subheader("Adversarial predictions (top-3)")
                for _class, desc, prob in decoded_adv:
                    st.write(f"- {desc} ({prob:.4f})")

                # convert arrays back to displayable images (uint8)
                orig_disp = deprocess(prep)[0]   # 224x224 uint8
                adv_disp  = deprocess(adv_np)[0]

                # perturbation visualization (magnified)
                perturb = adv_disp.astype(int) - orig_disp.astype(int)
                perturb_vis = np.clip(perturb * 10 + 128, 0, 255).astype(np.uint8)

                # Show Original, Noise (perturbation), Adversarial side-by-side:
                st.subheader("Result images (224 × 224)")
                c1, c2, c3 = st.columns([1, 1, 1])
                # Use a fixed small width so images stay reasonably sized
                img_width = 300
                with c1:
                    st.image(orig_disp, caption="Original", use_container_width=False, width=img_width)
                with c2:
                    st.image(perturb_vis, caption="Perturbation (x10)", use_container_width=False, width=img_width)
                with c3:
                    st.image(adv_disp, caption=f"Adversarial (eps={eps})", use_container_width=False, width=img_width)
                    # Download button placed directly under the adversarial image
                    adv_pil = Image.fromarray(adv_disp)
                    buf = io.BytesIO()
                    adv_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button("Download adversarial (PNG)", data=byte_im,
                                       file_name="adversarial.png", mime="image/png")

                # metrics (SSIM & PSNR)
                st.subheader("Perceptual metrics")
                try:
                    ssim_val = ssim(orig_disp, adv_disp, channel_axis=-1, win_size=7)
                except Exception:
                    ssim_val = ssim(orig_disp, adv_disp, channel_axis=-1)
                psnr_val = psnr(orig_disp, adv_disp)
                st.write(f"- SSIM: {ssim_val:.4f}")
                st.write(f"- PSNR: {psnr_val:.2f} dB")

else:
    st.info("Upload an image to begin.")

st.markdown("---")
st.caption("FGSM = Fast Gradient Sign Method. App uses MobileNetV2 (ImageNet).")
