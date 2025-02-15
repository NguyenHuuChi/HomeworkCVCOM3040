import numpy as np
import cv2

import numpy as np

def compute_gaussian_kernel(kernel_size: int, sigma=1):
    """
    Given an odd kernel_size (3, 5, 7, etc.) and sigma value,
    create a Gaussian kernel. Normalize the kernel before returning.
    """
    # 1) Ensure kernel_size is odd
    assert kernel_size % 2 == 1, "Kernel size must be an odd number"
    
    # 2) Create a meshgrid centered at 0
    k = kernel_size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    
    # 3) Compute the Gaussian function
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # 4) Normalize so sum of all elements = 1
    kernel /= np.sum(kernel)
    
    return kernel


def convolve(img, kernel):
    """
    Convolve function for odd dimensions.
    IT CONVOLVES IMAGES
    """
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")

    img_height = img.shape[0]
    img_width = img.shape[1]
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    
    pad = ((pad_height, pad_height), (pad_height, pad_width))
    g = np.empty(img.shape, dtype=np.float64)
    img = np.pad(img, pad, mode='constant', constant_values=0)

    #Do convolution
    for i in np.arange(pad_height, img_height+pad_height):
        for j in np.arange(pad_width, img_width+pad_width):
            roi = img[i - pad_height:i + pad_height +
                      1, j - pad_width:j + pad_width + 1]
            g[i - pad_height, j - pad_width] = (roi*kernel).sum()

    if (g.dtype == np.float64):
        kernel = kernel / 255.0
        kernel = (kernel*255).astype(np.uint8)
    else:
        g = g + abs(np.amin(g))
        g = g / np.amax(g)
        g = (g*255.0)
    
    return g

def cv2_imshow(img, title:str=""):
    """
    Display BGR image loaded from cv2.imread()
    It works when running local script file (.py) only
    """
    # display image with figure title
    cv2.imshow(title, img)

    # Wait for any key to be pressed before continuing, or we can specifiy a delay
    # for how long you keep the window open (in ms)
    cv2.waitKey(0)

    # Closes all open windows
    cv2.destroyAllWindows()