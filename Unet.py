import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras import layers

def attention_gate(x, g, filters):

    g1 = layers.Conv2D(filters, (1, 1), padding='same')(g)
    g1 = layers.ReLU()(g1)

    x1 = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x1 = layers.ReLU()(x1)

    merge = layers.Add()([x1, g1])
    merge = layers.ReLU()(merge)

    attention = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(merge)

    return layers.Multiply()([x, attention])


# Build U-Net Model——Config1
def build_unet_model(input_size): 
    inputs = layers.Input(input_size)

    # 编码器（下采样）
    c0 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c0 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c0)
    p0 = layers.MaxPooling2D((2, 2))(c0)

    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p0)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)


    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c4], axis=-1)  # Ensure concatenate happens along the channel axis
    u6 = attention_gate(u6, c4, 512)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c3], axis=-1)
    u7 = attention_gate(u7, c3, 256)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.concatenate([u8, c2], axis=-1)
    u8 = attention_gate(u8, c2, 128) 
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.concatenate([u9, c1], axis=-1)
    u9 = attention_gate(u9, c1, 64) 
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    u10 = layers.UpSampling2D((2, 2))(c9)
    u10 = layers.concatenate([u10, c0], axis=-1)
    u10 = attention_gate(u10, c0, 64)
    c10 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u10)
    c10 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c10)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c10)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

