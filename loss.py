import tensorflow as tf
import numpy as np

def edge_aware_smoothness_loss(image_orig, depth):  # 边缘感知平滑损失

    # 计算深度图在x和y方向的梯度 (一阶差分近似)
    grad_depth_y, grad_depth_x = np.gradient(depth)   # 计算深度在垂直(y)和水平(x)方向的梯度
    # 计算图像在x和y方向的梯度，以用于平滑权重
    grad_image_y, grad_image_x = np.gradient(image_orig)
    # 计算图像梯度幅值的绝对值，用于权重（这里采用逐像素的指数衰减权重）———— 应该是固定的(因为输入图像不会变，变的是predicted depth map)
    weight_x = np.exp(-np.abs(grad_image_x))
    weight_y = np.exp(-np.abs(grad_image_y))
    # 计算加权后的深度梯度绝对值
    smoothness_x = np.abs(grad_depth_x) * weight_x
    smoothness_y = np.abs(grad_depth_y) * weight_y
    # 平滑损失是水平和垂直方向平滑度的平均值(反映图像的平均平滑度)
    loss = np.mean(smoothness_x + smoothness_y)
    return loss

def approximate_geometric_consistency_loss(depth_a, depth_b):  # 近似几何一致性损失(用于相邻帧计算)
    # 计算尺度对齐系数：使用深度中值比将深度A对齐到深度B的尺度
    scale = np.median(depth_b) / (np.median(depth_a) + 1e-8)
    # 将深度A按比例缩放，使其尺度与深度B接近
    depth_a_aligned = depth_a * scale
    # 计算两个深度图的逐像素差异（这里假设相邻帧场景差异较小或已对准）
    diff = depth_a_aligned - depth_b
    # 取绝对值并求平均，得到深度不一致性的度量
    loss = np.mean(np.abs(diff))
    return loss

def feature_metric_loss(image_orig, depth):  # 特征匹配损失
    # 计算图像和深度的梯度分量
    grad_image_y, grad_image_x = np.gradient(image_orig)
    grad_depth_y, grad_depth_x = np.gradient(depth)
    # 计算图像和深度的梯度幅值 (使用L2范数，即 sqrt(dx^2 + dy^2))
    edge_image = np.sqrt(grad_image_x**2 + grad_image_y**2)
    edge_depth = np.sqrt(grad_depth_x**2 + grad_depth_y**2)
    # 计算梯度幅值图之间的差异
    diff = edge_image - edge_depth
    # 取绝对值并求平均，得到特征匹配损失
    loss = np.mean(np.abs(diff))
    return loss

def local_contrast_loss(y_orig, y_pred, window_size=3):
    """
    计算生成图像与目标图像之间的局部对比度差异
    :param y_true: 目标图像
    :param y_pred: 生成图像
    :param window_size: 局部对比度计算的窗口大小，通常为3x3或5x5
    :return: 局部对比度损失
    """
    # 计算目标图像和生成图像的局部标准差（局部对比度）
    mean_orig = tf.nn.avg_pool(y_orig, ksize=[1, window_size, window_size, 1],
                               strides=[1, 1, 1, 1], padding='SAME')
    squared_diff_orig = tf.square(y_orig - mean_orig)
    local_stddev_orig = tf.sqrt(tf.nn.avg_pool(squared_diff_orig, ksize=[1, window_size, window_size, 1],
                                               strides=[1, 1, 1, 1], padding='SAME'))

    mean_pred = tf.nn.avg_pool(y_pred, ksize=[1, window_size, window_size, 1],
                               strides=[1, 1, 1, 1], padding='SAME')
    squared_diff_pred = tf.square(y_pred - mean_pred)
    local_stddev_pred = tf.sqrt(tf.nn.avg_pool(squared_diff_pred, ksize=[1, window_size, window_size, 1],
                                               strides=[1, 1, 1, 1], padding='SAME'))

    # 计算目标图像和生成图像之间的局部对比度差异
    contrast_loss = tf.reduce_mean(tf.abs(local_stddev_orig - local_stddev_pred))
    return contrast_loss

def custom_loss(y_orig_and_true, y_pred):
    y_orig = y_orig_and_true[..., 0:1]  # 第1通道是原始图像
    y_true = y_orig_and_true[..., 1:2]  # 第2通道是GT深度图

    # 转换成同类型(否则在ssim计算时会报错)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # print("y_true 形状:", y_true.shape)
    # print("y_pred 形状:", y_pred.shape)

    # Define Weights for each Losses(L1权重设置大于edges)
    w_ssim = 0.8
    w_l1 = 10.0
    w_l2 = 8.0
    w_edges = 0.5
    w_lc = 0.5
    alpha = 0.1

    # Structural Similarity Index (SSIM) Loss
    # SSIM表示图像结构相似性，指标越大，结构越相似，且范围∈[0,1]
    # 为了定义成Loss，需要反转一下，故：Loss_SSIM = 1 - SSIM
    ssim_loss = tf.reduce_mean(
        1 - tf.image.ssim(
            y_orig, y_pred,
            max_val=1, filter_size=3, k1=0.01 ** 2, k2=0.03 ** 2
        )
    )  # max_val:计算图像的像素最大值(这里由于进行了Normalization，所以max是1)

    # L1 Loss(表征光度一致性损失)
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # L2 Loss
    l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Edge Loss
    # 1.计算两张图在x、y方向的梯度
    dy_true, dx_true = tf.image.image_gradients(y_true)
    # dy_orig, dx_orig = tf.image.image_gradients(y_orig)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)


    # 根据参考图像(y_orig or y_true)的梯度生成权重
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_orig)))
    # weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_orig)))

    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y
    edges_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

    # local_contrast_loss
    lc_loss = local_contrast_loss(y_orig, y_pred, window_size = 3.0)

    # Final Loss
    loss = (ssim_loss * w_ssim) + (l1_loss * w_l1) + (edges_loss * w_edges)
    # loss = (ssim_loss * w_ssim) + (l1_loss * w_l1) + (edges_loss * w_edges) + (lc_loss * w_lc)
    # loss = alpha * 0.5 * ssim_loss + (1 - alpha) * l1_loss
    return loss