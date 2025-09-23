import tensorflow as tf
import numpy as np

def edge_aware_smoothness_loss(image_orig, depth): 

    grad_depth_y, grad_depth_x = np.gradient(depth) 
    grad_image_y, grad_image_x = np.gradient(image_orig)
    weight_x = np.exp(-np.abs(grad_image_x))
    weight_y = np.exp(-np.abs(grad_image_y))
    smoothness_x = np.abs(grad_depth_x) * weight_x
    smoothness_y = np.abs(grad_depth_y) * weight_y
    loss = np.mean(smoothness_x + smoothness_y)
    return loss

def approximate_geometric_consistency_loss(depth_a, depth_b): 
    scale = np.median(depth_b) / (np.median(depth_a) + 1e-8)
    depth_a_aligned = depth_a * scale
    diff = depth_a_aligned - depth_b
    loss = np.mean(np.abs(diff))
    return loss

def feature_metric_loss(image_orig, depth):
    grad_image_y, grad_image_x = np.gradient(image_orig)
    grad_depth_y, grad_depth_x = np.gradient(depth)
    edge_image = np.sqrt(grad_image_x**2 + grad_image_y**2)
    edge_depth = np.sqrt(grad_depth_x**2 + grad_depth_y**2)
    diff = edge_image - edge_depth
    loss = np.mean(np.abs(diff))
    return loss

def local_contrast_loss(y_orig, y_pred, window_size=3):
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

    contrast_loss = tf.reduce_mean(tf.abs(local_stddev_orig - local_stddev_pred))
    return contrast_loss

def custom_loss(y_orig_and_true, y_pred):
    y_orig = y_orig_and_true[..., 0:1] 
    y_true = y_orig_and_true[..., 1:2]  
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    w_ssim = 0.8
    w_l1 = 10.0
    w_l2 = 8.0
    w_edges = 0.5
    w_lc = 0.5
    alpha = 0.1

    ssim_loss = tf.reduce_mean(
        1 - tf.image.ssim(
            y_orig, y_pred,
            max_val=1, filter_size=3, k1=0.01 ** 2, k2=0.03 ** 2
        )
    ) 

    # L1 Loss--Photographic
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # L2 Loss
    l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Edge Loss
    dy_true, dx_true = tf.image.image_gradients(y_true)
    # dy_orig, dx_orig = tf.image.image_gradients(y_orig)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)

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
