import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

from captum.attr import LayerGradCam, Saliency
from captum.attr import visualization as vit
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ==========================================
# PAGE CONFIG & CSS
# ==========================================
st.set_page_config(
    page_title="Deepfake Detective",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Glassmorphism background and text colors */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 1rem;
        color: #a1a1aa;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f4f4f5;
    }
    .verdict-real {
        color: #10b981; /* emerald-500 */
        font-weight: bold;
        font-size: 2.2rem;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
    }
    .verdict-fake {
        color: #ef4444; /* red-500 */
        font-weight: bold;
        font-size: 2.2rem;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL DEFINITIONS & HELPERS
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Autoencoder
    ae = Autoencoder().to(device)
    if os.path.exists("autoencoder_real.pth"):
        ae.load_state_dict(torch.load("autoencoder_real.pth", map_location=device))
    ae.eval()

    # 2. CNN
    cnn = models.efficientnet_b0()
    num_ftrs = cnn.classifier[1].in_features
    cnn.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, 2)
    )
    cnn = cnn.to(device)
    if os.path.exists("cnn_efficientnet.pth"):
        cnn.load_state_dict(torch.load("cnn_efficientnet.pth", map_location=device))
    cnn.eval()

    return ae, cnn, device

@st.cache_data
def get_cnn_config():
    # From notebook preprocessing logic
    return {
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "real_class_idx": 0,
        "fake_class_idx": 1
    }

ae_model, cnn_model, device = load_models()
cnn_cfg = get_cnn_config()

# Transforms
ae_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

cnn_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(cnn_cfg["norm_mean"], cnn_cfg["norm_std"]),
])

# Haar cascades for face crop
CASCADE_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_XML)

# ==========================================
# APP LOGIC
# ==========================================
st.title("🕵️‍♂️ Deepfake Detective: Unmasking the Truth")
st.markdown("##### *Is it an authentic portrait or a synthetic fabrication? Upload an image and let the neural ensemble decide.*")

# Sidebar
st.sidebar.header("Ensemble Settings")
use_face_crop = st.sidebar.checkbox("🔒 Auto Face Detection/Crop run prior to model", value=False)
cnn_weight = st.sidebar.slider("CNN Weighting", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="Higher favors CNN over the Autoencoder")
ae_threshold = st.sidebar.number_input("Autoencoder Anomaly Threshold", value=0.02, format="%.4f")

uploaded_file = st.file_uploader("Upload an Image...", type=["png", "jpg", "jpeg", "webp", "avif"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    if img_bgr is None:
        st.error("Error reading the image file.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            
            # Optionally Crop
            display_img = img_bgr.copy()
            if use_face_crop:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    display_img = img_bgr[y:y+h, x:x+w]
                    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Face Detected in Full Image", use_column_width=True)

            pil_img = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            if not use_face_crop:
                st.image(pil_img, caption="Original View", use_column_width=True)
            else:
                 st.image(pil_img, caption="Cropped Face to analyze", use_column_width=True)
                
        with col2:
            st.markdown("### 🔍 Analysis Results")
            if st.button("🚀 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing with Autoencoder & CNN..."):
                    
                    ae_in = ae_tf(pil_img).unsqueeze(0).to(device)
                    cnn_in = cnn_tf(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # AE
                        ae_out = ae_model(ae_in)
                        ae_err = torch.mean((ae_out - ae_in) ** 2).item()
                        ae_real_score = 1.0 if ae_err < ae_threshold else 0.0
                        
                        # CNN
                        cnn_out = cnn_model(cnn_in)
                        probs = torch.softmax(cnn_out, dim=1)[0]
                        prob_real = probs[cnn_cfg["real_class_idx"]].item()
                        prob_fake = probs[cnn_cfg["fake_class_idx"]].item()
                    
                    final_score = (cnn_weight * prob_real) + ((1 - cnn_weight) * ae_real_score)
                    verdict_text = "REAL" if final_score > 0.5 else "FAKE"
                    verdict_class = "verdict-real" if verdict_text == "REAL" else "verdict-fake"
                    
                    # Layout Metrics
                    st.markdown(f"<div style='text-align:center; margin-top:20px;'><span class='{verdict_class}'>Verdict: {verdict_text}</span></div>", unsafe_allow_html=True)
                    st.progress(final_score, text=f"Combined Reality Score: {final_score:.2%}")
                    
                    # Detailed metric cards
                    c_m1, c_m2 = st.columns(2)
                    with c_m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">CNN 'Fake' Prob</div>
                            <div class="metric-value">{prob_fake:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">AE Reconstruction Err</div>
                            <div class="metric-value">{ae_err:.5f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Save state for XAI
                    st.session_state["analyzed_image"] = pil_img
                    st.session_state["analyzed"] = True

        st.divider()

        # XAI Techniques
        if st.session_state.get("analyzed", False):
            st.markdown("### 🧠 Explainable AI (XAI) Vision")
            st.markdown("Unlock the neural black box to see exactly what region the CNN is focusing on.")
            
            if st.button("🔮 Generate XAI Insights", use_container_width=True):
                with st.spinner("Calculating gradients and building visual overlays..."):
                    img_for_xai = st.session_state["analyzed_image"]
                    img_np_224 = np.array(img_for_xai.resize((224, 224)))
                    xai_in = cnn_tf(img_for_xai).unsqueeze(0).to(device)
                    xai_in.requires_grad = True
                    
                    # Ensure models are in eval
                    cnn_model.eval()
                    target_layer = cnn_model.features[-1]
                    target_cls = cnn_cfg["real_class_idx"]
                    
                    # -- Grad-CAM --
                    gc = LayerGradCam(cnn_model, target_layer)
                    attr_gc = gc.attribute(xai_in, target=target_cls)
                    upsampled_attr = LayerGradCam.interpolate(attr_gc, (224, 224))
                    heatmap_gc, _ = vit.visualize_image_attr(
                        upsampled_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                        img_np_224 / 255.0,
                        method="blended_heat_map",
                        sign="all",
                        show_colorbar=True,
                        title="Grad-CAM",
                        use_pyplot=False
                    )
                    
                    # -- Saliency --
                    saliency = Saliency(cnn_model)
                    attr_sl = saliency.attribute(xai_in, target=target_cls)
                    heatmap_sl, _ = vit.visualize_image_attr(
                        attr_sl[0].cpu().permute(1, 2, 0).detach().numpy(),
                        img_np_224 / 255.0,
                        method="blended_heat_map",
                        sign="absolute_value",
                        show_colorbar=True,
                        title="Saliency Map",
                        use_pyplot=False
                    )

                    c_x1, c_x2 = st.columns(2)
                    with c_x1:
                        st.markdown("**Grad-CAM (Attention Heatmap)**")
                        st.pyplot(heatmap_gc)
                        st.info("Grad-CAM highlights the broad regions the CNN considers most important for its decision. Warmer colors = higher importance.")
                    with c_x2:
                        st.markdown("**Saliency Map (Pixel-Level Gradients)**")
                        st.pyplot(heatmap_sl)
                        st.info("Saliency maps trace back the prediction to the exact pixels. Helps identify granular, localized artifacts (like blending edges).")

                    c_x3, c_x4 = st.columns(2)
                    with c_x3:
                        st.markdown("**SHAP (Game-Theoretic Feature Importance)**")
                        background = torch.zeros(1, 3, 224, 224).to(device)
                        explainer = shap.GradientExplainer(cnn_model, background)
                        shap_values = explainer.shap_values(xai_in)
                        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
                        test_numpy = np.swapaxes(np.swapaxes(xai_in.cpu().detach().numpy(), 1, -1), 1, 2)
                        
                        shap.image_plot(shap_numpy, test_numpy, show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
                        st.info("SHAP explains the output by computing the exact game-theoretic contribution of each pixel grouping to the deepfake prediction.")
                    
                    with c_x4:
                        st.markdown("**LIME (Local Interpretable Model-Agnostic Explanations)**")
                        lime_explainer = lime_image.LimeImageExplainer()
                        
                        def batch_predict(images):
                            tf_lime = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(cnn_cfg["norm_mean"], cnn_cfg["norm_std"]),
                            ])
                            batch = torch.stack([tf_lime(Image.fromarray(i)) for i in images], dim=0).to(device)
                            logits = cnn_model(batch)
                            return F.softmax(logits, dim=1).detach().cpu().numpy()

                        explanation = lime_explainer.explain_instance(img_np_224, batch_predict, top_labels=2, hide_color=0, num_samples=100)
                        label_for_map = target_cls if target_cls in explanation.top_labels else explanation.top_labels[0]
                        temp, mask = explanation.get_image_and_mask(label_for_map, positive_only=True, num_features=5, hide_rest=False)

                        fig_lime, ax_lime = plt.subplots(figsize=(5, 5))
                        ax_lime.imshow(mark_boundaries(temp / 255.0, mask))
                        ax_lime.axis("off")
                        st.pyplot(fig_lime)
                        st.info("LIME perturbs the input to see what 'superpixels' influence the model the most, displaying the critical segments.")

else:
    st.session_state["analyzed"] = False
