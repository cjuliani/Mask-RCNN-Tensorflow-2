import io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_grid_and_boxes(background, gt_boxes, anchors=None, scores=None,
                       grid_points=None, pos_min_score=0.5):
    """Returns a figure with plotted ground truth and anchor
    boxes and/or the point grid specifying the location of
    anchors.

    Args:
        background (tensor): the background tensor to draw
            boxes in.
        gt_boxes (tensor): coordinates of the ground truth
            boxes.
        anchors (tensor): coordinates of the anchor bounding
            boxes.
        scores (tensor): the object scores associated to the
            anchor bounding boxes.
        grid_points (tensor): the anchor grid.
        pos_min_score (float): minimum object score for positive
            anchor boxes.
    """
    # Add background.
    fig, ax = plt.subplots()
    ax.imshow(background)

    # Display ground truth boxes.
    for coord in gt_boxes:
        top_left = (coord[0], coord[1])
        width = coord[2] - coord[0]
        height = coord[3] - coord[1]
        rect = patches.Rectangle(
            xy=top_left,
            width=width,
            height=height,
            linewidth=1,
            edgecolor='lime',
            facecolor='none')
        ax.add_patch(rect)

    # Display positive anchors.
    if anchors is not None:
        pos_boxes = tf.gather_nd(anchors, tf.where(tf.greater_equal(scores, pos_min_score)))
        for coord in pos_boxes:
            top_left = (coord[0], coord[1])
            width = coord[2] - coord[0]
            height = coord[3] - coord[1]
            rect = patches.Rectangle(
                xy=top_left,
                width=width,
                height=height,
                linewidth=1,
                edgecolor='r',
                facecolor='none')
            ax.add_patch(rect)

    # Display anchor grid.
    if grid_points is not None:
        for pos in grid_points:
            plt.plot(pos[0], pos[1], 'yo', markersize=1.2)

    plt.xticks([])
    plt.yticks([])
    return fig


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure'
    to a PNG image and returns it. The supplied figure is
    closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    io_buf = io.BytesIO()
    figure.savefig(io_buf, format='raw')

    # Closing the figure prevents it from being displayed
    # directly inside the notebook.
    io_buf.seek(0)

    # Convert PNG buffer to tensorflow image.
    width = int(figure.bbox.bounds[3])
    height = int(figure.bbox.bounds[2])
    return tf.reshape(tf.io.decode_raw(
        input_bytes=io_buf.getvalue(),
        out_type=tf.uint8), (width, height, 4))


def image_to_figure(image, cmap="jet"):
    """Converts an image to a figure with color gradient
    and returns it.

    Args:
        image: the image to convert.
        cmap: the color map.
    """
    figure = plt.figure(figsize=(1, 1))
    plt.xticks([])
    plt.yticks([])
    _ = plt.imshow(image, cmap=cmap)
    plt.close('all')
    return figure
