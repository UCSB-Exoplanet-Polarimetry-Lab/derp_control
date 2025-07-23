from katsu.katsu_math import np
from scipy import ndimage
from scipy.optimize import minimize
from skimage import feature, filters


def fit_circle_to_beam(image, threshold=None, edge_method='canny'):
    """
    Fit a circle to the beam by detecting its circular boundary.

    Parameters:
    -----------
    image : numpy.ndarray
        2D array containing the beam image
    threshold : float, optional
        Threshold for creating binary mask. If None, auto-determined
    edge_method : str
        Edge detection method: 'canny', 'gradient', or 'threshold'

    Returns:
    --------
    center : tuple
        (y, x) coordinates of fitted circle center
    radius : float
        Radius of fitted circle
    edge_points : numpy.ndarray
        Edge points used for fitting
    """

    if threshold is None:
        # Use a more conservative threshold for edge detection
        threshold = np.mean(image) + 0.3 * np.std(image)

    if edge_method == 'canny':
        # Use Canny edge detection for clean edges
        edges = feature.canny(image, sigma=1.0, low_threshold=threshold*0.3,
                             high_threshold=threshold*0.8)
    elif edge_method == 'gradient':
        # Use gradient-based edge detection
        grad_mag = filters.sobel(image)
        edges = grad_mag > (np.mean(grad_mag) + np.std(grad_mag))
    else:  # threshold method
        # Simple threshold-based edge detection
        binary = image > threshold
        edges = binary ^ ndimage.binary_erosion(binary, iterations=2)

    # Get edge points
    edge_points = np.column_stack(np.where(edges))

    if len(edge_points) < 10:
        raise ValueError("Not enough edge points found. Try adjusting the threshold.")

    # Fit circle using least squares optimization
    def circle_residuals(params, points):
        """Calculate residuals for circle fitting"""
        center_y, center_x, radius = params
        distances = np.sqrt((points[:, 0] - center_y)**2 + (points[:, 1] - center_x)**2)
        return distances - radius

    def objective(params, points):
        """Objective function for circle fitting"""
        residuals = circle_residuals(params, points)
        return np.sum(residuals**2)

    # Initial guess: center of mass and mean distance
    initial_center = np.mean(edge_points, axis=0)
    distances = np.sqrt(np.sum((edge_points - initial_center)**2, axis=1))
    initial_radius = np.mean(distances)

    initial_guess = [initial_center[0], initial_center[1], initial_radius]

    # Optimize
    result = minimize(objective, initial_guess, args=(edge_points,))

    if not result.success:
        print("Warning: Circle fitting may not have converged properly")

    center_y, center_x, radius = result.x

    return (center_y, center_x), radius, edge_points


def center_beam_by_circle(image, threshold=None, edge_method='canny'):
    """
    Center a beam image based on fitted circle geometry.

    Parameters:
    -----------
    image : numpy.ndarray
        2D array containing the beam image
    threshold : float, optional
        Threshold for edge detection
    edge_method : str
        Edge detection method

    Returns:
    --------
    centered_image : numpy.ndarray
        Image with beam centered based on circle fit
    shift : tuple
        (dy, dx) shift applied
    circle_params : dict
        Dictionary containing fitted circle parameters
    """

    # Fit circle to beam
    (center_y, center_x), radius, edge_points = fit_circle_to_beam(
        image, threshold, edge_method)

    # Calculate shift needed to center the circle
    image_center = np.array(image.shape) / 2
    fitted_center = np.array([center_y, center_x])
    shift = image_center - fitted_center

    # Apply the shift
    centered_image = ndimage.shift(image, shift, order=1, mode='constant', cval=0)

    circle_params = {
        'center': (center_y, center_x),
        'radius': radius,
        'edge_points': edge_points,
        'fit_quality': calculate_fit_quality(edge_points, (center_y, center_x), radius)
    }

    return centered_image, shift, circle_params


def calculate_fit_quality(edge_points, center, radius):
    """Calculate how well the circle fits the edge points"""
    center_y, center_x = center
    distances = np.sqrt((edge_points[:, 0] - center_y)**2 + (edge_points[:, 1] - center_x)**2)
    residuals = distances - radius
    rmse = np.sqrt(np.mean(residuals**2))
    return rmse


def robust_circle_fit(image, methods=['canny', 'gradient'], thresholds=None):
    """
    Try multiple methods to find the best circle fit.
    """
    if thresholds is None:
        base_thresh = np.mean(image) + 0.3 * np.std(image)
        thresholds = [base_thresh * 0.7, base_thresh, base_thresh * 1.3]

    best_fit = None
    best_quality = float('inf')

    for method in methods:
        for thresh in thresholds:
            try:
                centered, shift, params = center_beam_by_circle(image, thresh, method)
                quality = params['fit_quality']

                if quality < best_quality:
                    best_quality = quality
                    best_fit = (centered, shift, params)

            except (ValueError, Exception) as e:
                print(f"Method {method} with threshold {thresh:.3f} failed: {e}")
                continue

    if best_fit is None:
        raise ValueError("Could not fit circle with any method")

    return best_fit
