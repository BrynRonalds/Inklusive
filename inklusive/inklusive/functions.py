# Functions needed for image recommendations

import skimage.io
from skimage.transform import resize
from PIL import Image


def image_transform(test_img_path, shape_resize):
    """
    Resize and normalize the input image
    """
    image = skimage.io.imread(test_img_path, as_gray=False)
    img_resized = resize(image, shape_resize,
                         anti_aliasing=True,
                         preserve_range=True)
    assert img_resized.shape == shape_resize
    img_resized = img_resized / 255.0
    return img_resized
