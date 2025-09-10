"""Just some utilities for drawing masks on images
"""
from katsu.katsu_math import np
from prysm.coordinates import make_xy_grid, cart_to_polar
import ipdb

def create_circular_aperture(shape, radius=1, center=(0, 0)):
    """
    Create a circular mask with a given radius.

    Parameters:
    -----------
    shape : tuple or int
        Shape of the mask (height, width)
    radius : float
        Radius of the circle (in pixels)
    center : list or tuple
        Center of the circle (y, x) in pixels. Default is [0, 0] (center of the mask).

    Returns:
    --------
    numpy.ndarray
        2D boolean array with True inside the circle and False outside
    """
    if not isinstance(shape, (tuple, list)) and not isinstance(shape, int):
        raise ValueError("Shape must be a tuple or list of two integers (height, width)")

    # unpack center
    cy, cx = center
    x = np.arange(shape)
    if x.ndim == 1:
        y, x = np.meshgrid(x, x)
    mask_bool = ((x - cx)**2 + (y - cy)**2) <= radius**2

    return mask_bool

def create_circular_obscuration(shape, radius=1, center=(0, 0)):
    """
    Create a circular obscuration mask with a given radius.

    Parameters:
    -----------
    shape : tuple or int
        Shape of the mask (height, width)
    radius : float
        Radius of the circle (in pixels)
    center : list or tuple
        Center of the circle (y, x) in pixels. Default is [0, 0] (center of the mask).

    Returns:
    --------
    numpy.ndarray
        2D boolean array with False inside the circle and True outside
    """
    mask = create_circular_aperture(shape, radius, center)

    return ~mask  # Invert the mask to create an obscuration

if __name__ == "__main__":
    # Example usage
    data = np.random.rand(100, 100)  # Example data
    mask = create_circular_aperture(shape=(100, 100), radius=1, center=(0, 0))
    mask_inverted = create_circular_obscuration(shape=(100, 100), radius=0.1, center=(0, 0))
    # You can visualize the mask using matplotlib or any other library
    import matplotlib.pyplot as plt
    plt.imshow(data * mask * mask_inverted, cmap='gray')
    plt.title('Circular Mask')
    plt.show()
