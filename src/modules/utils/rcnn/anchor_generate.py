import tensorflow as tf
from src.modules.utils.rcnn import bbox_ops


@tf.function(input_signature=(
        tf.TensorSpec(shape=None, dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.float32)))
def get_anchors_corners(image_width, image_height, anchor_scales, anchor_ratios, stride):
    """Returns the coordinates of anchors boxes. The number
    boxes per anchor depends on the number of anchor scales
    and ratios.

    Args:
        image_width (float): the width of image from which
            bounding boxes were estimated.
        image_height (float): the height of image from which
            bounding boxes were estimated.
        anchor_scales (tensor): the anchor scale coefficients.
        anchor_ratios (tensor): the anchor ratio coefficients.
        stride (float): the stride considered to determine
            anchors in the image.
    """
    anchor_corner_offsets = generate_anchors(anchor_scales, anchor_ratios)  # (9, 4)

    # Get dimensions (w,h) given <image_width> and <stride>
    # Here, stride is 16 because VGG applies 4 max-pooling. So,
    # the width of grid cell corresponding to respective region in
    # input image is image_width / (stride * 4).
    grid_cell_width = tf.math.ceil(image_width / stride)   # 1024/16=64
    grid_cell_height = tf.math.ceil(image_height / stride)
    total_pos = tf.cast(grid_cell_height * grid_cell_width, tf.int32)   # number of cells; 64**2=4096

    # Get (x,y) center position of each cell.
    offset_x = (tf.range(grid_cell_width) * stride) + stride//2  # (64,)
    offset_y = (tf.range(grid_cell_height) * stride) + stride//2

    # Build corresponding mesh grid.
    x, y = tf.meshgrid(offset_x, offset_y)  # coord mesh grid; (64, 64)
    x = tf.reshape(x, [-1])  # (4096,)
    y = tf.reshape(y, [-1])
    coordinates = tf.stack((x, y, x, y), axis=-1)  # (4096, 4)
    coordinates = tf.transpose(tf.reshape(coordinates, [1, total_pos, 4]), (1, 0, 2))  # (4096, 1, 4)

    # Calculate box corners (9 boxes, 4 coordinates) per
    # grid cell.
    anchor_corners = coordinates + anchor_corner_offsets  # (4096, 9, 4)
    anchor_corners = tf.reshape(anchor_corners, [-1, 4])  # (4096 * 9, 4)

    # Clip boxes
    anchor_corners = bbox_ops.clip_head_boxes(anchor_corners, image_width, image_height)
    return tf.cast(anchor_corners, tf.float32), tf.transpose(tf.stack([x, y]))


def generate_anchors(anchor_scales, anchor_ratios, anchor_bias_x_ctr=0, anchor_bias_y_ctr=0):
    """Returns the corner positions of anchors.

    Args:
        anchor_scales (tensor): the anchor scale coefficients.
        anchor_ratios (tensor): the anchor ratio coefficients.
        anchor_bias_x_ctr (int): pixel shifting over x from
            image origin.
        anchor_bias_y_ctr (int): pixel shifting over y from
            image origin.
    """
    anchor_scales_mat = tf.stack((anchor_scales, anchor_scales), axis=-1)
    anchor_size = ratios_process(anchor_scales_mat, anchor_ratios)
    return corners_coordinates(anchor_size, anchor_bias_x_ctr, anchor_bias_y_ctr)


def ratios_process(anchor_scales, anchor_ratios):
    """Returns anchors window sizes based on anchor scales
    and ratios."""
    anchor_area = tf.cast(anchor_scales[:, 0] * anchor_scales[:, 1], tf.float32)
    width = tf.round(tf.math.sqrt(tf.reshape(anchor_area, (-1, 1)) / anchor_ratios))
    height = width * anchor_ratios
    return tf.reshape(tf.stack((width, height), axis=-1), (-1, 2))  # (9, 2)


def corners_coordinates(anchor_size, x_ctr, y_ctr):
    """Returns the corner positions of anchors given
    the anchor sizes.

    Args:
        anchor_size (tensor): the anchors window sizes.
        x_ctr (int): pixel shifting over x from
            image origin.
        y_ctr (int): pixel shifting over y from
            image origin.
    """
    width = anchor_size[:, 0]
    height = anchor_size[:, 1]
    x1 = tf.round(x_ctr - 0.5 * width)  # -x pixels from origin
    y1 = tf.round(y_ctr - 0.5 * height)
    x2 = tf.round(x_ctr + 0.5 * width)  # +x pixels from origin
    y2 = tf.round(y_ctr + 0.5 * height)
    return tf.stack((x1, y1, x2, y2), axis=-1)
