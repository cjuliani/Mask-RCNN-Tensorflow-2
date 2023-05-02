import numpy as np
from scipy import ndimage


def rotate(array, angle):
    """Returns the input array rotated at a given angle."""
    return ndimage.rotate(input=array, angle=angle, order=0, reshape=False)


def scale(array, scaling, **kwargs):
    """Returns the input array scaled at the given scaling
    factor.

    Source:
        https://stackoverflow.com/questions/37119071/
    """
    h, w = array.shape[:2]

    # For multichannel images, do not apply the zoom factor to the RGB
    # dimension. Instead, create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions added to width and
    # height.
    zoom_tuple = (scaling,) * 2 + (1,) * (array.ndim - 2)

    # Zooming out.
    if scaling < 1:
        # Bounding box of the zoomed-out image within the output array.
        zh = int(np.round(h * scaling))
        zw = int(np.round(w * scaling))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding.
        out = np.zeros_like(array)
        out[top:top + zh, left:left + zw] = ndimage.zoom(array, zoom_tuple, order=0)

    # Zooming in.
    elif scaling > 1:
        # Bounding box of the zoomed-in region within the input array.
        zh = int(np.round(h / scaling))
        zw = int(np.round(w / scaling))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndimage.zoom(array[top:top + zh, left:left + zw], zoom_tuple, order=0)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges.
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    else:
        # If zoom_factor == 1, just return the input array
        out = array
    return out


# noinspection PyTypeChecker
def shift_pixels(array, size, direction, rate, **kwargs):
    """Returns the input array shifted within a given direction
    and at the specified rate.

    Args:
        array (ndarray): the input array to shift pixels from.
        size (int): the width or height of the array.
        direction (list): the directions of pixel shifting.
        rate (int): the reference rate factor of shifting,
            which is used to calculate the rate of shifting in
            pixels depending on the height or width of the input
            array.
    """
    # translate within the input limit range (converted in %)
    rate = int((size / 100) * rate)
    arr = np.zeros_like(array)
    # iterate procedure to each channel
    for c in range(array.shape[-1]):
        if type(direction) == list:
            translation = [i * rate for i in direction]
        else:
            translation = direction * rate
        arr[..., c] = ndimage.shift(array[..., c], translation, order=0)
    return arr


def flip_h(array):
    """Returns the input array flipped horizontally."""
    return np.fliplr(array)


def flip_v(array):
    """Returns the input array flipped vertically."""
    return np.flipud(array)


def do_nothing(array, **kwargs):
    return array


def augment_geometry(array, size, select, angle, scaling, direction, translation_rate):
    """Returns the input array with augmented geometry.

    Flip vertically or horizontally first, then shift pixels and
    finally scale and rotate the array. Parameters for augmentation
    are taken randomly. Some augmentation may not be performed.

    Args:
        array (ndarray): the input array to shift pixels from.
        angle (int): the angle at which the array is rotated.
        select (tuple or list): the selection indices taken randomly
            to specify which augmentation methods to apply.
        size (int): the width or height of the array.
        direction (int): the direction of shifting (-1 or 1).
        scaling (float): the scaling factor.
        translation_rate (int): the reference rate factor of shifting,
            which is used to calculate the rate of shifting in
            pixels depending on the height or width of the input
            array.
    """
    # (1) Apply flipping (or not).
    functions = [flip_h, flip_v, do_nothing]
    arr = functions[select[0]](array)

    # (2) Apply pixel shifting (or not).
    functions = [shift_pixels, do_nothing]
    arr = functions[select[1]](
        array=arr,
        scaling=scaling,
        direction=direction,
        rate=translation_rate,
        size=size)

    # (3) Apply scaling (or not).
    arr = scale(
        array=arr,
        scaling=scaling)

    # (4) rotate
    return rotate(arr, angle=angle)


def flip_and_rotate(array, select, angle):
    """Returns the input array with augmented geometry (flipping
    and rotation only).


    Args:
        array (ndarray): the input array to shift pixels from.
        angle (int): the angle at which the array is rotated.
        select (tuple or list): the selection indices taken randomly
            to specify which augmentation methods to apply.
    """
    # (1) Apply flipping (or not).
    functions = [flip_h, flip_v, do_nothing]
    arr = functions[select[0]](array)

    # (2) Apply rotation (or not).)
    return rotate(arr, angle=angle)
