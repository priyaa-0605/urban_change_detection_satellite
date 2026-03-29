import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import cv2
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# =====================================
# 1️⃣ DEVICE
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================
# 2️⃣ SIAMESE MODEL
# =====================================
class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        diff = torch.abs(f1 - f2)
        return self.decoder(diff)

# =====================================
# 3️⃣ LOAD MODEL
# =====================================
model = SiameseUNet().to(device)

model_path = "model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully")
else:
    print("❌ model.pth not found")

model.eval()

# =====================================
# 4️⃣ PREPROCESS
# =====================================
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    return image.to(device)

# =====================================
# 5️⃣ ROUTES
# =====================================

# 🌍 Landing Page (3D Earth)
@app.route("/")
def landing():
    return render_template("landing.html")

# 🧠 Main App Page
@app.route("/home")
def home():
    return render_template("index.html", result_image=False)

# =====================================
# 6️⃣ PREDICTION
# =====================================
@app.route("/predict", methods=["POST"])
def predict():

    before = request.files["before_image"]
    after = request.files["after_image"]

    upload_folder = "uploads"
    static_folder = "static"

    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)

    before_path = os.path.join(upload_folder, "before.png")
    after_path = os.path.join(upload_folder, "after.png")

    before.save(before_path)
    after.save(after_path)

    # preprocess
    before_tensor = preprocess_image(before_path)
    after_tensor = preprocess_image(after_path)

    # =========================
    # MODEL INFERENCE
    # =========================
    with torch.no_grad():
        output = model(before_tensor, after_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()

    # normalize
    output = (output - output.min()) / (output.max() + 1e-8)

    # threshold (noise control 🔥)
    mask = (output > 0.3).astype(np.uint8)

    # =========================
    # REMOVE NOISE (IMPORTANT)
    # =========================
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # =========================
    # RESIZE
    # =========================
    after_img = cv2.imread(after_path)
    mask_resized = cv2.resize(mask, (after_img.shape[1], after_img.shape[0]))

    # =========================
    # OVERLAY
    # =========================
    overlay = after_img.copy()
    overlay[mask_resized == 1] = [255, 0, 0]

    final_overlay = cv2.addWeighted(after_img, 0.7, overlay, 0.3, 0)

    # save result
    result_path = os.path.join(upload_folder, "result.png")
    cv2.imwrite(result_path, final_overlay)

    # copy to static
    before_static = os.path.join(static_folder, "before.png")
    after_static = os.path.join(static_folder, "after.png")
    result_static = os.path.join(static_folder, "result.png")

    Image.open(before_path).save(before_static)
    Image.open(after_path).save(after_static)
    cv2.imwrite(result_static, final_overlay)

    # =========================
    # CHANGE %
    # =========================
    change_pixels = np.sum(mask)
    total_pixels = mask.size
    change_percentage = (change_pixels / total_pixels) * 100

    # =========================
    # CHANGE TYPE (LOGIC)
    # =========================
    if change_percentage < 2:
        change_type = "No Change"
    elif change_percentage < 5:
        change_type = "Minor Change"
    elif change_percentage < 15:
        change_type = "Moderate Change"
    else:
        change_type = "Major Change"

    # =========================
    # RESNET (PLACEHOLDER)
    # =========================
    # (You can replace this with actual ResNet model later)
    resnet_class = "Land 🌍"

    # =========================
    # AI EXPLANATION
    # =========================
    if change_percentage < 2:
        change_text = "No significant structural changes detected."
    elif change_percentage < 5:
        change_text = "Minor variations observed, possibly small modifications."
    elif change_percentage < 15:
        change_text = "Moderate development or environmental change detected."
    else:
        change_text = "Significant structural or urban expansion detected."

    return render_template(
        "index.html",
        result_image=True,
        change_percent=round(change_percentage, 2),
        change_type=change_type,
        resnet_class=resnet_class,
        change_text=change_text
    )

# =====================================
# 7️⃣ RUN
# =====================================
if __name__ == "__main__":
    app.run(debug=True)