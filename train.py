# System
import time
import os

# Functions
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2


# Machine Learning
import tensorflow as tf
from tensorflow.keras import layers, models
import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
from Unet import build_unet_model
from loss import custom_loss
from sklearn.model_selection import train_test_split

def lr_scheduler(epoch, lr):
    if epoch < 3:
        return lr  # Keep the initial learning rate for the first 3 epochs
    else:
        return lr * tf.math.exp(-0.1)  # Exponential decay after the third epoch

batch_size_value = 8

def batch_generator(image_paths, depth_paths, batch_size=batch_size_value, image_size=(256, 256)):
    while True:
        # batch_images = []
        # batch_depths = []

        batch_images = []
        batch_images_and_depths = []

        for i in range(batch_size):
            index = np.random.randint(0, len(image_paths))
            image_path = image_paths[index]
            depth_path = depth_paths[index]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

            # Resize images and depth maps to image_size
            image = cv2.resize(image, image_size)
            depth = cv2.resize(depth, image_size)

            # max pixel value
            max_image_value = np.max(image)
            # print("max_image_value", max_image_value)
            max_depth_value = np.max(depth)
            # print("max_depth_value", max_depth_value)

            # Normalize images and depth maps
            image = image.astype('float32') / max_image_value
            max_image_value_normalized = np.max(image)
            # print("max_image_value_normalized", max_image_value_normalized)
            depth = depth.astype('float32') / max_depth_value  # Depth normalization
            max_depth_value_normalized = np.max(depth)
            # print("max_depth_value_normalized", max_depth_value_normalized)

            image = np.expand_dims(image, axis=-1)  
            depth = np.expand_dims(depth, axis=-1) 

            # batch_images.append(image)
            # batch_depths.append(depth)

            batch_images.append(image)

            combined = np.concatenate([image, depth], axis=-1)  # shape = (H, W, 2)
            batch_images_and_depths.append(combined)

        yield np.array(batch_images), np.array(batch_images_and_depths)

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
image_folder = 'data/endo_train_exvivo/p1_grayscale'
depth_folder = 'data/endo_train_exvivo/p1_depth_grayscale' 

train_rgb = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]  # jpg
train_depth = [os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith('.png')]  # png

train_rgb_paths, val_rgb_paths, train_depth_paths, val_depth_paths = train_test_split(
    train_rgb, train_depth, test_size=0.1, random_state=42
)

unet_model = build_unet_model(input_size=(256, 256, 1))  # 输入尺寸

# unet_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

unet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=custom_loss,
    metrics=['mae']
)

# Callback setup
lr_scheduler_callback = LearningRateScheduler(lr_scheduler, verbose=1)

early_stopping = EarlyStopping(
    # monitor="val_loss",
    monitor="loss",
    mode="min",
    verbose=1,
    patience=4
)

model_checkpoint = ModelCheckpoint(
    "./model/exvivo/pig_lung/best_unet_model.h5",
    monitor="loss",
    mode="min",
    verbose=1,
    save_best_only=True
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    # update_freq='batch',
    update_freq=1,
    write_graph=True,
    write_images=True
)

callbacks = [early_stopping, model_checkpoint, lr_scheduler_callback, tensorboard_callback]

train_generator = batch_generator(train_rgb_paths, train_depth_paths, batch_size=batch_size_value, image_size=(256, 256))
val_generator = batch_generator(val_rgb_paths, val_depth_paths, batch_size=batch_size_value, image_size=(256, 256))

# Training the model
start_time = time.time()

tf.keras.backend.clear_session()

with tf.device('/GPU:0'):  # Using GPU
    unet_history = unet_model.fit(
        train_generator,
        steps_per_epoch=len(train_rgb_paths) // batch_size_value,
        validation_data=val_generator,
        validation_steps=len(val_rgb_paths) // batch_size_value,
        epochs=4,
        callbacks=callbacks,

        # validation_split = 0.1,  
        # shuffle = True  
    )

end_time = time.time()

unet_time = end_time-start_time

print("Time:", unet_time)

loss = unet_history.history['loss']
val_loss = unet_history.history['val_loss']

# Create a DataFrame for easy plotting
df = pd.DataFrame({
    'Epochs': range(1, len(loss) + 1),
    'Training Loss': loss,
    'Validation Loss': val_loss
})

# Set seaborn style
sns.set(style="whitegrid")

# Create line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Epochs', y='Training Loss', marker='o', label='Training Loss')
sns.lineplot(data=df, x='Epochs', y='Validation Loss', marker='o', label='Validation Loss')

# Set labels and title
plt.xlabel('Epochs')
plt.ylabel('Custom Loss')
plt.title('Training and Validation Loss')

# Show the plot

plt.show()
