import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras import layers

def attention_gate(x, g, filters):
    """
    实现注意力门控机制。
    :param x: 解码器的输出特征图
    :param g: 跳跃连接的特征图
    :param filters: 卷积滤波器的数量
    :return: 通过注意力机制加权后的特征图
    """
    # 1x1卷积对跳跃连接进行处理
    g1 = layers.Conv2D(filters, (1, 1), padding='same')(g)
    g1 = layers.ReLU()(g1)

    # 1x1卷积对解码器输出进行处理
    x1 = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x1 = layers.ReLU()(x1)

    # 将两者加起来
    merge = layers.Add()([x1, g1])
    merge = layers.ReLU()(merge)

    # 生成注意力系数（sigmoid激活生成[0, 1]的系数）
    attention = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(merge)

    # 将输入特征图与注意力系数相乘，增强重要区域
    return layers.Multiply()([x, attention])


# Build U-Net Model——Config1
def build_unet_model(input_size):  # 修改输入尺寸为512x512
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

    # 底部层（瓶颈）
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # 解码器（上采样）
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c4], axis=-1)  # Ensure concatenate happens along the channel axis
    u6 = attention_gate(u6, c4, 512)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c3], axis=-1)
    u7 = attention_gate(u7, c3, 256)  # 在此加上注意力机制
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.concatenate([u8, c2], axis=-1)
    u8 = attention_gate(u8, c2, 128)  # 在此加上注意力机制
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.concatenate([u9, c1], axis=-1)
    u9 = attention_gate(u9, c1, 64)  # 在此加上注意力机制
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    u10 = layers.UpSampling2D((2, 2))(c9)
    u10 = layers.concatenate([u10, c0], axis=-1)
    u10 = attention_gate(u10, c0, 64)  # 在此加上注意力机制
    c10 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u10)
    c10 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c10)

    # 输出层
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c10)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# # Build U-Net Model——Config2
# def build_unet_model(input_size):
#     # Inputs(根据不同输入图像的规模要调整)
#     inputs = layers.Input(input_size)
#
#     # Encoder: Downsample
#     conv1, pool1 = downsample_block(inputs, 64)
#     conv2, pool2 = downsample_block(pool1, 128)
#     conv3, pool3 = downsample_block(pool2, 256)
#     conv4, pool4 = downsample_block(pool3, 512)
#
#     # Bottleneck
#     bottleneck = double_conv_block(pool4, 1024)
#
#     # Decoder: Upsample
#     up6 = upsample_block(bottleneck, conv4, 512)
#     up7 = upsample_block(up6, conv3, 256)
#     up8 = upsample_block(up7, conv2, 128)
#     up9 = upsample_block(up8, conv1, 64)
#
#     # Outputs
#     outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(up9)
#
#     # Model
#     unet_model = models.Model(inputs, outputs, name="U-Net")
#
#     return unet_model
#
#
# def double_conv_block(x, n_filters):
#     x = tf.keras.layers.Conv2D(n_filters, (3, 3), padding="same", activation="relu")(x)
#     x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#
#     x = tf.keras.layers.Conv2D(n_filters, (3, 3), padding="same", activation="relu")(x)
#     x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     return x
#
#
# def downsample_block(x, n_filters):
#     conv = double_conv_block(x, n_filters)
#     pool = tf.keras.layers.MaxPool2D((2, 2))(conv)
#     return conv, pool
#
#
# def upsample_block(x, conv, n_filters):
#     x = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding="same")(x)
#     x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#
#     x = tf.keras.layers.concatenate([x, conv])
#
#     x = double_conv_block(x, n_filters)
#     return x
