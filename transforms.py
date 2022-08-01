import tensorflow as tf
from keras.layers.preprocessing import image_preprocessing as image_ops
import math
import numpy as np


def to_4d(image: tf.Tensor) -> tf.Tensor:
    """Converts an input Tensor to 4 dimensions.
  4D image => [N, H, W, C] or [N, C, H, W]
  3D image => [1, H, W, C] or [1, C, H, W]
  2D image => [1, H, W, 1]
  Args:
    image: The 2/3/4D input tensor.
  Returns:
    A 4D image tensor.
  Raises:
    `TypeError` if `image` is not a 2/3/4D tensor.
  """
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
    """Converts a 4D image back to `ndims` rank."""
    shape = tf.shape(image)
    begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def wrap(image: tf.Tensor) -> tf.Tensor:
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf.shape(image)
    extended_channel = tf.expand_dims(tf.ones(shape[:-1], image.dtype), -1)
    extended = tf.concat([image, extended_channel], axis=-1)
    return extended


def unwrap(image: tf.Tensor, replace: float) -> tf.Tensor:
    """Unwraps an image produced by wrap.
  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.
  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.
  Returns:
    image: A 3D image Tensor with 3 channels.
  """
    image_shape = tf.shape(image)
    # Flatten the spatial dimensions.
    flattened_image = tf.reshape(image, [-1, image_shape[-1]])

    # Find all pixels where the last channel is zero.
    alpha_channel = tf.expand_dims(flattened_image[..., 3], axis=-1)

    # replace = tf.concat([[replace], tf.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    flattened_image = tf.where(
        tf.equal(alpha_channel, 0),
        tf.ones_like(flattened_image, dtype=image.dtype) * replace,
        flattened_image)

    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(
        image,
        [0] * image.shape.rank,
        tf.concat([image_shape[:-1], [3]], -1))
    return image


def _convert_angles_to_transform(angles: tf.Tensor, image_width: tf.Tensor,
                                 image_height: tf.Tensor) -> tf.Tensor:
    """Converts an angle or angles to a projective transform.
  Args:
    angles: A scalar to rotate all images, or a vector to rotate a batch of
      images. This must be a scalar.
    image_width: The width of the image(s) to be transformed.
    image_height: The height of the image(s) to be transformed.
  Returns:
    A tensor of shape (num_images, 8).
  Raises:
    `TypeError` if `angles` is not rank 0 or 1.
  """
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    if len(angles.get_shape()) == 0:  # pylint:disable=g-explicit-length-test
        angles = angles[None]
    elif len(angles.get_shape()) != 1:
        raise TypeError('Angles should have a rank 0 or 1.')
    x_offset = ((image_width - 1) -
                (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) *
                 (image_height - 1))) / 2.0
    y_offset = ((image_height - 1) -
                (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) *
                 (image_height - 1))) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(
        values=[
            tf.math.cos(angles)[:, None],
            -tf.math.sin(angles)[:, None],
            x_offset[:, None],
            tf.math.sin(angles)[:, None],
            tf.math.cos(angles)[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.dtypes.float32),
        ],
        axis=1,
    )


def _convert_translation_to_transform(translations: tf.Tensor) -> tf.Tensor:
    """Converts translations to a projective transform.
  The translation matrix looks like this:
    [[1 0 -dx]
     [0 1 -dy]
     [0 0 1]]
  Args:
    translations: The 2-element list representing [dx, dy], or a matrix of
      2-element lists representing [dx dy] to translate for each image. The
      shape must be static.
  Returns:
    The transformation matrix of shape (num_images, 8).
  Raises:
    `TypeError` if
      - the shape of `translations` is not known or
      - the shape of `translations` is not rank 1 or 2.
  """
    translations = tf.convert_to_tensor(translations, dtype=tf.float32)
    if translations.get_shape().ndims is None:
        raise TypeError('translations rank must be statically known')
    elif len(translations.get_shape()) == 1:
        translations = translations[None]
    elif len(translations.get_shape()) != 2:
        raise TypeError('translations should have rank 1 or 2.')
    num_translations = tf.shape(translations)[0]

    return tf.concat(
        values=[
            tf.ones((num_translations, 1), tf.dtypes.float32),
            tf.zeros((num_translations, 1), tf.dtypes.float32),
            -translations[:, 0, None],
            tf.zeros((num_translations, 1), tf.dtypes.float32),
            tf.ones((num_translations, 1), tf.dtypes.float32),
            -translations[:, 1, None],
            tf.zeros((num_translations, 2), tf.dtypes.float32),
        ],
        axis=1,
    )


def translate(image: tf.Tensor, translations) -> tf.Tensor:
    """Translates image(s) by provided vectors.
  Args:
    image: An image Tensor of type uint8.
    translations: A vector or matrix representing [dx dy].
  Returns:
    The translated version of the image.
  """
    transforms = _convert_translation_to_transform(translations)
    return transform(image, transforms=transforms)


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
    """Blend image1 and image2 using 'factor'.
  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 1.
  Args:
    image1: An image Tensor of type float32.
    image2: An image Tensor of type float32.
    factor: A floating point value above 0.0.
  Returns:
    A blended image Tensor of type float32.
  """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if 0.0 < factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.float32)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 1.0), tf.float32)


def transform(image: tf.Tensor, transforms) -> tf.Tensor:
    """Prepares input data for `image_ops.transform`."""
    original_ndims = tf.rank(image)
    transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    if transforms.shape.rank == 1:
        transforms = transforms[None]
    image = to_4d(image)
    image = image_ops.transform(
        images=image, transforms=transforms, interpolation='nearest')
    return from_4d(image, original_ndims)


def identity(img, magnitude=None):
    return img


def solarize(image: tf.Tensor, threshold: float = .5) -> tf.Tensor:
    """Solarize the input image(s)."""
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 1 from the pixel.
    return tf.where(image < threshold, image, image - 1)


def autocontrast(image: tf.Tensor, magnitude=None) -> tf.Tensor:
    """Implements Autocontrast function from PIL using TF ops.
  Args:
    image: A 3D float32 tensor.
  Returns:
    The image after it has had autocontrast applied to it and will be of type float32.
  """

    def scale_channel(image: tf.Tensor) -> tf.Tensor:
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 1.
        def scale_values(im):
            scale = 1.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 1.0)
            return tf.cast(im, tf.float32)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[..., 0])
    s2 = scale_channel(image[..., 1])
    s3 = scale_channel(image[..., 2])
    image = tf.stack([s1, s2, s3], -1)

    return image * 255


def posterize(image: tf.Tensor, bits: float) -> tf.Tensor:
    """Equivalent of PIL Posterize."""

    shift = 8 - int(8 * (1 - bits))

    image = tf.cast(image * 255.0, tf.uint8)

    image = tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)

    return tf.cast(image, tf.float32)


def translate_x(image: tf.Tensor, fraction: float, replace: float = 0.0) -> tf.Tensor:
    """Equivalent of PIL Translate in X dimension."""
    pixels = image.shape[0] * fraction
    image = translate(wrap(image), [-pixels, 0])
    return unwrap(image, replace)


def translate_y(image: tf.Tensor, fraction: float, replace: float = 0.0) -> tf.Tensor:
    """Equivalent of PIL Translate in Y dimension."""
    pixels = image.shape[0] * fraction
    image = translate(wrap(image), [0, -pixels])
    return unwrap(image, replace)


def shear_x(image: tf.Tensor, level: float, replace: float = 0.0) -> tf.Tensor:
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    image = transform(
        image=wrap(image), transforms=[1., level, 0., 0., 1., 0., 0., 0.])
    return unwrap(image, replace)


def shear_y(image: tf.Tensor, level: float, replace: float = 0.0) -> tf.Tensor:
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    image = transform(
        image=wrap(image), transforms=[1., 0., 0., level, 1., 0., 0., 0.])
    return unwrap(image, replace)


def color(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Color."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return blend(degenerate, image, factor)


def contrast(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Contrast."""
    image = tf.cast(255.0 * image, tf.dtypes.uint8)
    degenerate = tf.image.rgb_to_grayscale(image)
    # Cast before calling tf.histogram.
    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256, dtype=tf.dtypes.int32)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.float32))
    return tf.cast(blend(degenerate, image, 1 - factor), tf.float32)


def brightness(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image) if np.random.choice([True, False], 1) else tf.ones_like(image)
    return blend(degenerate, image, 1 - factor)


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = tf.cast(image * 255, tf.uint8)
    image = tf.cast(image * 255, tf.float32)
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = tf.constant(
        [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
        shape=[3, 3, 1, 1]) / 13.
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(
        image, kernel, strides, padding='VALID', dilations=[1, 1])
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return tf.cast(blend(result, orig_image, 1 - factor), tf.float32)


def equalize(image: tf.Tensor, magnitude=None) -> tf.Tensor:
    """Implements Equalize function from PIL using TF ops."""

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[..., c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(
            tf.equal(step, 0), lambda: im,
            lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.float32)

    image = tf.cast(255.0 * image, tf.dtypes.uint8)
    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], -1)
    return tf.cast(image, tf.dtypes.float32)


def rotate(image: tf.Tensor, range: float) -> tf.Tensor:
    """Rotates the image by degrees either clockwise or counterclockwise.
  Args:
    image: An image Tensor of type float32.
    range: Float, a scalar angle in [0, 1] to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
  Returns:
    The rotated version of image.
  """
    range = range if np.random.choice([True, False], 1) else 0 - range
    degrees = range * 360
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = tf.cast(degrees * degrees_to_radians, tf.float32)

    original_ndims = tf.rank(image)
    image = to_4d(image)

    image_height = tf.cast(tf.shape(image)[1], tf.float32)
    image_width = tf.cast(tf.shape(image)[2], tf.float32)
    transforms = _convert_angles_to_transform(
        angles=radians, image_width=image_width, image_height=image_height)
    # In practice, we should randomize the rotation degrees by flipping
    # it negatively half the time, but that's done on 'degrees' outside
    # of the function.
    image = transform(image, transforms=transforms)
    return from_4d(image, original_ndims)


def rand_augment_object(M, N, leq_M=False):
    """
    :param M: global magnitude parameter over all transformations (1 is max, 0 is min)
    :param N: number of transformations to apply
    :param leq_M: perform transformations less than or equal to M
    :return: a callable that will randomly augment a single image (a function from image space to image space)
    """

    def rand_augment(img):
        """
        performs random augmentation with magnitude M for N iterations

        :param img: image to augment
        :return: augmented image
        """
        transforms = [identity, autocontrast, equalize, rotate, solarize, color, posterize, contrast, brightness,
                      sharpness, shear_x, shear_y, translate_x, translate_y]
        # needs to take a rank 3 numpy tensor, and return a tensor of the same rank
        for op in np.random.choice(transforms, N):
            img = op(img, np.random.uniform(0, M)) if leq_M else op(img, M)
        return img

    return rand_augment


def custom_rand_augment_object(M, N, leq_M=False):
    """
    :param M: global magnitude parameter over all transformations (1 is max, 0 is min)
    :param N: number of transformations to apply
    :param leq_M: perform transformations less than or equal to M
    :return: a callable that will randomly augment a single image (a function from image space to image space)
    """

    def rand_augment(img):
        """
        performs random augmentation with magnitude M for N iterations

        :param img: image to augment
        :return: augmented image
        """
        transforms = [identity, autocontrast, equalize, rotate, color, contrast, brightness,
                      sharpness, shear_x, shear_y, translate_x, translate_y]
        # needs to take a rank 3 numpy tensor, and return a tensor of the same rank
        for op in np.random.choice(transforms, N):
            img = op(img, np.random.uniform(0, M)) if leq_M else op(img, M)
        return img

    return rand_augment
