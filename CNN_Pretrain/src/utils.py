import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import random_shapes
import logging


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def namestr(obj, namespace=globals()):
    """
    a = 'some var'
    namestr(a, globals())
    ['a']
    """
    return [name for name in namespace if namespace[name] is obj]


def plot_enc_layer(n, x_enc):
    for i in range(n):
        # display encoded
        ax = plt.subplot(1, n, i + 1)
        plt.title("encoded images")
        plt.imshow(x_enc[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_bw_figs(n, x_enc, x_rec, x_orig):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        bx = plt.subplot(3, n, i + n + 1)
        plt.title("original")
        plt.imshow(x_orig[i])
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)

        # display encoded
        ax = plt.subplot(3, n, i + 1)
        plt.title("encoded images")
        plt.imshow(x_enc[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(3, n, i + 2*n + 1)
        plt.title("reconstructed")
        plt.imshow(x_rec[i])
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()


def generate_target(obstacles):
    shape_range = (1, 5)
    coverage_range = (0.2, 0.8)
    area = np.product(self.shape)

    target = __generate_random_shapes_area(
        shape_range[0],
        shape_range[1],
        area * coverage_range[0],
        area * coverage_range[1]
    )

    return target & ~obstacles


def __generate_random_shapes(self, min_shapes, max_shapes):
    img, _ = random_shapes(self.shape, max_shapes, min_shapes=min_shapes, multichannel=False,
                           allow_overlap=True, random_seed=np.random.randint(2**32 - 1))
    # Numpy random usage for random seed unifies random seed which can be set for repeatability
    attempt = np.array(img != 255, dtype=bool)
    return attempt, np.sum(attempt)


def __generate_random_shapes_area(self, min_shapes, max_shapes, min_area, max_area, retry=100):
    for attemptno in range(retry):
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
        if min_area is not None and min_area > area:
            continue
        if max_area is not None and max_area < area:
            continue
        return attempt
    print("Here")
    logging.warning("Was not able to generate shapes with given area constraint in allowed number of tries."
                    " Randomly returning next attempt.")
    attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
    logging.warning("Size is: ", area)
    return attempt