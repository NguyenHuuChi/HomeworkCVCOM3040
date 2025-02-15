import numpy as np
import cv2
from utils import compute_gaussian_kernel, convolve, cv2_imshow

sobel_kx = np.array(None, dtype=np.float32) #TODO: complete the sobel_kx kernel
sobel_ky = np.array(None, dtype=np.float32) #TODO: complete the sobel_ky kernel
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
def harris(img, threshold=0.25, k=0.04):

    """
    Complete the missing steps for a full implementation.
    
    Input:
        img: Input image in color or grayscale.
        threshold (0-1): Threshold for detecting corners.
        k: an empirical constanst, typically k=0.04

    Output:
        img_cpy: Image with corners marked.
    """

    img_cpy = img.copy() # copying image

    #TODO:
    """
        1. Convert image to gray scale
        2. Compute image gradients Ix, Iy using Sobel kernels
        3. Compute the products of gradients Ixx, Iyy, Ixy at each pixel
        4. Apply Gaussian smooth for the derivative products
        5. Compute the Harris response and normalize to (0-1)
        6. Thresholding the response
    """

    #Step 1-5: Your code here
    image_gray = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2GRAY)
    Ix = convolve(image_gray, sobel_kx)
    Iy = convolve(image_gray, sobel_ky)
    Ixx=Ix**2
    Iyy=Iy**2
    Ixy=Ix*Iy
    
    gauss_kernel = compute_gaussian_kernel(kernel_size=3, sigma=1.0)
    Ixx = convolve(Ixx, gauss_kernel)
    Iyy = convolve(Iyy, gauss_kernel)
    Ixy = convolve(Ixy, gauss_kernel)
    
    R = (Ixx * Iyy - Ixy ** 2) - k * (Ixx + Iyy) ** 2
    
    # Normalize R to [0, 1] for easier thresholding (avoid divide-by-zero if image is flat).
    R_min, R_max = R.min(), R.max()
    if R_max - R_min == 0:
        # If the image has no variation, no corners can be found
        return img_cpy  
    R_norm = (R - R_min) / (R_max - R_min)
    # Step 6: Thresholding the response
    loc = np.where(R_norm >= threshold) #TODO: Find all points above the threshold
    # drawing filtered points
    for pt in zip(*loc[::-1]):
        cv2.circle(img_cpy, pt, 2, (0, 0, 255), -1)

    return img_cpy

def main():
    """
    Load the image, apply the Harris function, and visualize the results.
    """
    # 1. Define the image path (Adjust to your actual file location)
    image_path = r"D:\Computer-Vision-2025-Jan-25_19-39-22-456\Computer-Vision-2025-Jan-25_19-39-22-456\viewer\files\lab02 assignment\lab02 assignment\lab02_code\test_image.jpg"
    
    # 2. Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image at path: {image_path}")
    
    # 3. Apply the Harris corner detection
    #    Adjust the threshold or k value as needed for best results
    threshold = 0.25
    k_value = 0.04
    corners_marked = harris(img, threshold=threshold, k=k_value)
    
    # 4. Show the result
    #    If you have a custom cv2_show function, you can use that instead.
    cv2.imshow("Harris Corners", corners_marked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: Save the output if desired
    output_path = "harris_corners_result.jpg"
    cv2.imwrite(output_path, corners_marked)
    print(f"Processed image saved at: {output_path}")
# Run the program
if __name__ == "__main__":
    main()