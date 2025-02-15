import numpy as np
import cv2
from utils import compute_gaussian_kernel, convolve, cv2_imshow
def apply_sobel_filters(img):
    """
    Args:
    - img: Input image (2D numpy array).

    Returns:
    - G: Gradient magnitude (scaled to 0-255).
    - theta: Gradient direction (radians, range [-pi, pi]).
    """
    # Define Sobel kernels
    sobel_kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # Convolve the image with both Sobel kernels
    img_x = convolve(img, sobel_kx)
    img_y = convolve(img, sobel_ky)

    # Compute the gradient magnitude
    G = np.sqrt(img_x**2 + img_y**2)

    # Scale G to the range 0-255
    min_val, max_val = np.min(G), np.max(G)
    # Avoid division by zero in case the image is uniform
    if max_val - min_val == 0:
        G = np.zeros_like(G)
    else:
        G = (G - min_val) / (max_val - min_val) * 255

    # Compute the gradient direction with arctan2
    theta = np.arctan2(img_y, img_x)

    return G, theta

def non_maximal_suppression(img, gradient_angle): # gradient_angle is in radians
    """
    Apply Non maximum supression to thin out the edges
    Args:
        - gradient_angle: theta resulted from apply_sobel_filters function
    """
    img_out = np.zeros((img.shape[0],img.shape[1])) # image after non max suppression
    angle_degrees = gradient_angle * 180. / np.pi
    angle_degrees[angle_degrees < 0] += 180
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            prev_pixel = 0 # previous pixel in the direction of gradient
            next_pixel = 0
            # close to horizontal
            if (0 <= angle_degrees[i,j] < 22.5) or (157.5 <= angle_degrees[i,j]<= 180):
                prev_pixel = img[i, j-1] # i is row, j is column
                next_pixel = img[i, j+1]
            # close to 45 degrees
            elif (22.5 <= angle_degrees[i,j] < 67.5):
                prev_pixel = img[i+1, j-1]
                next_pixel = img[i-1, j+1]
            #close to 90 degrees
            elif (67.5 <= angle_degrees[i,j] < 112.5):
                prev_pixel = img[i+1, j]
                next_pixel = img[i-1, j]
            #close to 135 degrees
            elif (112.5 <= angle_degrees[i,j] < 157.5):
                prev_pixel = img[i-1, j-1]
                next_pixel = img[i+1, j+1]
            # compare previous and next pixels in the gradient direction and
            # zero out current pixel if it is smaller than the prev. or next.
            if (img[i,j] >= prev_pixel) and (img[i,j] >= next_pixel):
                img_out[i,j] = img[i,j]
            else:
                img_out[i,j] = 0
    return img_out

def double_threshold_image(img, lowThreshold=20, highThreshold=80, weak_value=50):
    """
    Apply double thresholding to the image based on the provided thresholds.

    Args:
    - img: Input image (2D numpy array).
    - lowThreshold: Lower threshold value.
    - highThreshold: Upper threshold value.
    - weak_value: Value to assign to pixels in the weak range.

    Returns:
    - thresholded_img: Output image after applying double thresholding.
    """
    W, H = img.shape  # Get the dimensions of the input image
    thresholded_img = np.zeros((W, H), dtype=np.int32)  # Initialize the output image with zeros

    strong_value = 255  # The value for "strong" pixels
    
    # Determine the indices of strong, weak, and zero-value pixels
    # Use np.where() with appropriate condition checks for weak, strong, and zero thresholds
    # 1. Pixels below lowThreshold → Zero
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    # 2. Pixels between lowThreshold and highThreshold → Weak
    weak_i, weak_j = np.where((img >= lowThreshold) & (img < highThreshold))
    
    # 3. Pixels above highThreshold → Strong
    strong_i, strong_j = np.where(img >= highThreshold)


    # Assign strong, weak, and zero values to the thresholded image
    thresholded_img[strong_i, strong_j] = strong_value  # Assign strong_value to strong pixels
    thresholded_img[weak_i, weak_j] = weak_value  # Assign weak_value to weak pixels
    thresholded_img[zeros_i, zeros_j] = 0  # Assign 0 to the rest
    
    return thresholded_img

def hysteresis(img, weak, strong=255):
    """
    Perform hysteresis to link weak pixels to strong ones if connected.

    Args:
    - img: Input thresholded image.
    - weak: Value representing weak pixels.
    - strong: Value representing strong pixels (default: 255).

    Returns:
    - img: Image after applying hysteresis to link weak pixels to strong ones.
    """
    for i in range(1, img.shape[0] - 1):  # Traverse the image, avoiding the borders
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == weak:  # Check if the pixel is weak
                # Check if any of the 8 neighboring pixels are strong
                # TODO: Complete the condition to check neighboring strong pixels
                if (img[i-1, j-1] == strong) or (img[i, j-1] == strong) or \
                   (img[i+1, j-1] == strong) or (img[i-1, j] == strong) or \
                   (img[i+1, j] == strong) or (img[i-1, j+1] == strong) or \
                   (img[i, j+1] == strong) or (img[i+1, j+1] == strong):
                    img[i, j] = strong  # Convert weak pixel to strong
                else:
                    img[i, j] = 0  # Set isolated weak pixels to 0
    return img

def main():
    """
    Main function to perform edge detection using the following steps:
    1. Load an image.
    2. Convert it to grayscale.
    3. Compute and apply a Gaussian kernel for smoothing.
    4. Apply Sobel filters to get gradient magnitudes and directions.
    5. Perform non-maximal suppression to thin out edges.
    6. Apply double thresholding to classify pixels as strong, weak, or zero.
    7. Use hysteresis to finalize edge detection by linking weak edges to strong ones.
    8. Display the final result (using cv2_show function)
    """
    # --- Step 0: Configuration ---
    low_threshold = 10
    high_threshold = 20
    weak_value = 15
    kernel_size = 5
    sigma = 1.0

    # --- Step 1: Load an image ---
    image_path = r"D:\Computer-Vision-2025-Jan-25_19-39-22-456\Computer-Vision-2025-Jan-25_19-39-22-456\viewer\files\lab02 assignment\lab02 assignment\lab02_code\test_image.jpg"
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # --- Step 2: Convert to grayscale ---
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- Step 3: Compute and apply Gaussian kernel for smoothing ---
    gauss_kernel = compute_gaussian_kernel(kernel_size, sigma) 
    # smoothed_image = convolve(image_gray, gauss_kernel)  # If using scipy.ndimage.convolve
    # Or, if you have your own convolve function:
    smoothed_image = convolve(image_gray, gauss_kernel)

    # --- Step 4: Apply Sobel filters to get gradient magnitude and direction ---
    grad_magnitude, grad_direction = apply_sobel_filters(smoothed_image)

    # --- Step 5: Non-maximal suppression to thin out edges ---
    nms_image = non_maximal_suppression(grad_magnitude, grad_direction)

    # --- Step 6: Apply double thresholding ---
    dt_image = double_threshold_image(nms_image, low_threshold, high_threshold, weak_value)

    # --- Step 7: Use hysteresis to finalize edge detection (link weak edges) ---
    final_edges = hysteresis(dt_image,weak_value)

    # --- Step 8: Display or save the result ---
    # cv2_show(final_edges, title="Final Edge Detection")
    # If you want to show with cv2.imshow instead:
    cv2.imshow("Final Edge Detection", final_edges.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    saved_path = "final_edges.jpg"
    cv2.imwrite(saved_path, final_edges.astype(np.uint8))
    print(f"Final edge-detected image saved to: {saved_path}")
    # Optionally, save the final result if desired:
    # cv2.imwrite("final_edges.jpg", final_edges)

# Run the program
if __name__ == "__main__":
    main()