from katsu.katsu_math import np

def bin_array_2d(array, bin_size, method='mean'):
    """
    Bin down a 2D numpy array using mean or median values.

    Parameters:
    -----------
    array : numpy.ndarray
        Input 2D array to be binned
    bin_size : int or tuple
        Size of the binning window. If int, same size used for both dimensions.
        If tuple, (height_bin, width_bin)
    method : str
        Binning method - 'mean' or 'median'

    Returns:
    --------
    numpy.ndarray
        Binned array with reduced dimensions
    """
    if isinstance(bin_size, int):
        bin_h, bin_w = bin_size, bin_size
    else:
        bin_h, bin_w = bin_size

    h, w = array.shape[0], array.shape[1]

    # Calculate new dimensions (truncate if doesn't divide evenly)
    new_h = h // bin_h
    new_w = w // bin_w

    # Trim array to make it evenly divisible
    trimmed_array = array[:new_h * bin_h, :new_w * bin_w]

    # Reshape to group pixels into bins
    reshaped = trimmed_array.reshape(new_h, bin_h, new_w, bin_w)

    # Apply binning method
    if method == 'mean':
        binned = np.nanmean(reshaped, axis=(1, 3))
    elif method == 'median':
        binned = np.nanmedian(reshaped, axis=(1, 3))
    else:
        raise ValueError("Method must be 'mean' or 'median'")

    return binned
