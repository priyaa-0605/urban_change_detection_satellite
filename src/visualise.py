import cv2
import matplotlib.pyplot as plt
import os

DATASET_PATH = "../data/train"

before_dir = os.path.join(DATASET_PATH, "A")
after_dir  = os.path.join(DATASET_PATH, "B")
label_dir  = os.path.join(DATASET_PATH, "label")

img_name = os.listdir(before_dir)[0]

before_img = cv2.imread(os.path.join(before_dir, img_name))
after_img  = cv2.imread(os.path.join(after_dir, img_name))
label_img  = cv2.imread(os.path.join(label_dir, img_name), 0)

before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
after_img  = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Before Image")
plt.imshow(before_img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("After Image")
plt.imshow(after_img)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Change Mask")
plt.imshow(label_img, cmap="gray")
plt.axis("off")

plt.show()
