import os
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
INPUT_PATH = r'c:\Users\youssefsaad5\Downloads\deep_fake_project\dataset\own_dataset2'
OUTPUT_PATH = r'c:\Users\youssefsaad5\Downloads\deep_fake_project\dataset\own_dataset_extracted2\real'
TARGET_SIZE = (512, 512)
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
MARGIN_PERCENT = 0.2  # 20% margin

def extract_faces():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # Clear existing files to avoid confusion
    for f in glob.glob(os.path.join(OUTPUT_PATH, '*')):
        os.remove(f)
        
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    
    image_files = glob.glob(os.path.join(INPUT_PATH, '*.jpeg')) + glob.glob(os.path.join(INPUT_PATH, '*.jpg'))
    print(f"Found {len(image_files)} images in {INPUT_PATH}")
    
    extracted_faces_to_vis = []
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Extracting faces")):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Using more robust parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            continue
            
        for f_idx, (x, y, w, h) in enumerate(faces):
            # Calculate margin
            mx = int(w * MARGIN_PERCENT)
            my = int(h * MARGIN_PERCENT)
            
            # Apply margin and clip to image boundaries
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(img.shape[1], x + w + mx)
            y2 = min(img.shape[0], y + h + my)
            
            # Crop face with margin
            face_crop = img[y1:y2, x1:x2]
            
            # Resize
            face_resized = cv2.resize(face_crop, TARGET_SIZE)
            
            # Naming convention: real_own_XX_Y.png
            filename = f"real_own_{idx:02d}_{f_idx}.png"
            output_file = os.path.join(OUTPUT_PATH, filename)
            
            # Save as PNG
            cv2.imwrite(output_file, face_resized)
            
            # Keep track for visualization (max 12 faces)
            if len(extracted_faces_to_vis) < 12:
                extracted_faces_to_vis.append((cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB), filename))
                
    # Visualization
    if extracted_faces_to_vis:
        plt.figure(figsize=(15, 10))
        n = len(extracted_faces_to_vis)
        cols = 4
        rows = (n + cols - 1) // cols
        for i, (face, name) in enumerate(extracted_faces_to_vis):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(face)
            plt.title(name)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(OUTPUT_PATH), 'visualization.png'))
        print(f"Visualization saved to {os.path.dirname(OUTPUT_PATH)}\\visualization.png")
        plt.show() # Note: This might not work in some environments, but we saved the file too.

if __name__ == "__main__":
    extract_faces()
