import cv2
import numpy as np

def apply_harris_operator(source: np.ndarray, k: float = 0.05) -> np.ndarray:

    # Convert to grayscale for calculations
    src = np.copy(source)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Get Gradient of image
    I_x = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=5)
    I_y = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=5)
    
    # Apply GaussianBlue for noise cancellation
    Ixx = cv2.GaussianBlur(src=I_x ** 2, ksize=(5, 5), sigmaX=0)
    Ixy = cv2.GaussianBlur(src=I_y * I_x, ksize=(5, 5), sigmaX=0)
    Iyy = cv2.GaussianBlur(src=I_y ** 2, ksize=(5, 5), sigmaX=0)

    det_H = Ixx * Iyy - Ixy ** 2
    trace_H = Ixx + Iyy
    harris_response = det_H - k * (trace_H ** 2)

    return harris_response


def map_indices_to_image(source: np.ndarray, harris_response: np.ndarray, threshold: float = 0.01) -> tuple:
    
    harris_copy = np.copy(harris_response)
    harris_matrix = cv2.dilate(harris_copy, None) ## Dilate the points to be more clear
    max_corner_response = np.max(harris_matrix)

    # Threshold for an optimal value, it may vary depending on the image.
    corner_indices = np.array(harris_matrix > (max_corner_response * threshold), dtype="int8")
    edges_indices = np.array(harris_matrix < 0, dtype="int8")
    flat_indices = np.array(harris_matrix == 0, dtype="int8")

    source = source[:corner_indices.shape[0], :corner_indices.shape[1]]
    source[corner_indices == 1] = [0, 0, 255]

    return source
