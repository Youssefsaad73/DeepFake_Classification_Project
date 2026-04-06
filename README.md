# Deepfake Detective: A Convolutional & Autoencoder Ensemble

This repository hosts a hybrid deep learning ensemble specifically designed to detect whether portrait images are authentic ("Real") or synthetically fabricated ("Fake"). It includes iterative data exploration, model training, analytical evaluation, and a fully functional interactive web portal.

## 🧠 Architecture Overview
The detection pipeline fuses two complementary techniques:
1. **Unsupervised Autoencoder:** Trained exclusively on Real images. It attempts to reconstruct portraits, calculating an anomaly score (Mean Squared Error) based on the structural reconstruction difference.
2. **Supervised EfficientNet-B0 CNN:** Fine-tuned on a heavily augmented dataset comprising Real and Fake datasets (including custom Haar Cascade extracted datasets) to predict the statistical probability of a forgery.

By blending the anomaly score and the CNN probability dynamically, the system acts as a highly robust discriminator capable of highlighting both broad synthetic textures and granular boundary artifacts.

## 🚀 Features
- **Dynamic Face Crop:** OpenCV Haar Cascade integration automatically isolates faces for processing prior to inference limit background noise.
- **Explainable AI (XAI):** The prediction isn't a black box. Our pipeline utilizes:
  - **Grad-CAM:** Identifies holistic regions of manipulation.
  - **Saliency Maps:** Traces back granular, pixel-level features causing predictions.
  - **SHAP:** Calculates precise game-theoretic feature importance.
  - **LIME:** Segments images to discover overriding super-pixels.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/deepfake-detective.git
   cd deepfake-detective
   ```

2. **Create a Virtual Environment:**
   If you use Anaconda/Miniconda:
   ```bash
   conda create -n deepfake_env python=3.8
   conda activate deepfake_env
   ```
   Or using standard Python:
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # On Linux/macOS
   .\.venv\Scripts\activate        # On Windows
   ```

3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application:**
   After installing the dependencies and ensuring your model weights (`*.pth`) are in the root directory:
   ```bash
   python -m streamlit run app.py
   ```
   *Note: If the application cannot find the weights, ensure you have ran through the Jupyter Notebooks to generate and save `autoencoder_real.pth` and `cnn_efficientnet.pth`!*

## 📁 Repository Structure
*   `optimized_*.ipynb`: The structured, iterative Jupyter Notebooks establishing data extraction, modeling, validation, and final ensemble tests.
*   `app.py`: The deployment-ready Streamlit frontend script representing the full downstream integration. 
*   `extract_faces_own.py`: Utility demonstrating standard facial cropping from general images via OpenCV.
*   `requirements.txt`: Environment package dependencies.
