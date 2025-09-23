# Import Libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import load_model
from loss import custom_loss
import random
import time

# Warning
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

# Structural F-score computing function
from Structural_F_score import compute_f_score

# data path
BASE_PATH = r"C:\Python_Program\Self_Supervised\Unet\data\endo_train_phantom"
TRAIN_PATH = os.path.join(BASE_PATH, "rgb_grayscale")  # training dataset
DEPTH_PATH = os.path.join(BASE_PATH, "depth_grayscale_c1")  # depth dataset

image_paths = [os.path.join(TRAIN_PATH, f) for f in os.listdir(TRAIN_PATH) if f.endswith('.jpg')]  # jpg
depth_paths = [os.path.join(DEPTH_PATH, f) for f in os.listdir(DEPTH_PATH) if f.endswith('.png')]  # png

# number consistent
assert len(image_paths) == len(depth_paths), "Not consistent！"

# load model
# unet_model = tf.keras.models.load_model('./model/exvivo/best_unet_model.h5', custom_objects={'custom_loss': custom_loss})
unet_model = tf.keras.models.load_model('./model/phantom/best_unet_model.h5', custom_objects={'custom_loss': custom_loss})


fig1, axs1 = plt.subplots(k, 3, figsize=(12, 5))
plt.suptitle("Outcome", x=0.55, y=0.93)

random_indices = random.sample(range(len(image_paths)), k)

for i, idx in enumerate(random_indices):
    rgb_path = image_paths[idx]
    depth_path = depth_paths[idx]

    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    # print("RGB Shape:", rgb_image.shape)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    depth_image = np.expand_dims(depth_image, -1)
    # print("Depth Shape:", depth_image.shape)

    image_size = (256, 256)

    rgb_image_resized = cv2.resize(rgb_image, image_size) / 255.0
    depth_image_resized = cv2.resize(depth_image, image_size) / 255.0

    rgb_image_resized = np.expand_dims(rgb_image_resized, axis=0)
    depth_image_resized = np.expand_dims(depth_image_resized, axis=0)

    start_time = time.time()

    predicted_depth = unet_model.predict(rgb_image_resized)

    end_time = time.time()

    inference_time = end_time - start_time
    print(f"[Image {i + 1}] Time: {inference_time:.4f} s")

    input_image_2d = rgb_image_resized.squeeze()
    gt_depth_2d = depth_image_resized.squeeze()
    pred_depth_2d = predicted_depth.squeeze()

    pred_depth_2d_save = np.uint8(pred_depth_2d * 255)
    cv2.imwrite(f'predicted_depth_image_{i}.png', pred_depth_2d_save)

    ''' SSIM Computing '''
    input_tensor = tf.convert_to_tensor(input_image_2d[np.newaxis, ..., np.newaxis], dtype=tf.float32)
    gt_tensor = tf.convert_to_tensor(gt_depth_2d[np.newaxis, ..., np.newaxis], dtype=tf.float32)
    pred_tensor = tf.convert_to_tensor(pred_depth_2d[np.newaxis, ..., np.newaxis], dtype=tf.float32)

    ssim_input_gt = tf.image.ssim(input_tensor, gt_tensor, max_val=1.0).numpy()[0]
    ssim_input_pred = tf.image.ssim(input_tensor, pred_tensor, max_val=1.0).numpy()[0]
    print(f"[Image {i + 1}] SSIM(Input vs GT): {ssim_input_gt:.4f} | SSIM(Input vs Pred): {ssim_input_pred:.4f}")

    ''' MSE and MAE '''
    mse = mean_squared_error(gt_depth_2d, pred_depth_2d)
    mae = mean_absolute_error(gt_depth_2d, pred_depth_2d)
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f}")

    ''' EAE '''
    edge_error = compute_edge_alignment_error(pred_depth_2d, gt_depth_2d)
    print(f"Edge Alignment Error (EAE) for Image {i + 1}: {edge_error:.4f}")

    rgb_filename = os.path.basename(rgb_path)
    depth_filename = os.path.basename(depth_path)

    axs1[i, 0].imshow(rgb_image_resized.squeeze(), cmap="gray")
    axs1[i, 0].axis('off')
    axs1[i, 0].set_title(f"RGB Image\n{rgb_filename}", fontsize=10)

    axs1[i, 1].imshow(depth_image_resized.squeeze(), cmap="magma")
    axs1[i, 1].axis('off')
    axs1[i, 1].set_title(f"Ground Truth Depth\n{depth_filename}", fontsize=10)

    axs1[i, 2].imshow(pred_depth_2d.squeeze(), cmap="magma")
    axs1[i, 2].axis('off')
    axs1[i, 2].set_title("Predicted Depth", fontsize=10)

plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, hspace=0.35, wspace=0.3)

plt.show()
