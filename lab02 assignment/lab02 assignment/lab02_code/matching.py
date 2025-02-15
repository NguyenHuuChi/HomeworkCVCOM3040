import cv2
import numpy as np

PATH_IMG = {
    "source": "matching_source.png",
    "template": "matching_template.png",
}



def main():
    """
    Main script
    Returns:
        A list of bounding boxes that satisfy the template matching.
        Displays the source image with bounding boxes drawn around matches.
    """
    # 1. Read the two images
    # img_source = cv2.imread("D:\Computer-Vision-2025-Jan-25_19-39-22-456\Computer-Vision-2025-Jan-25_19-39-22-456\viewer\files\lab02 assignment\lab02 assignment\lab02_code\matching_source.png")
    # img_template = cv2.imread("D:\Computer-Vision-2025-Jan-25_19-39-22-456\Computer-Vision-2025-Jan-25_19-39-22-456\viewer\files\lab02 assignment\lab02 assignment\lab02_code\matching_template.png")
    
    image_path = r"D:\Computer-Vision-2025-Jan-25_19-39-22-456\Computer-Vision-2025-Jan-25_19-39-22-456\viewer\files\lab02 assignment\lab02 assignment\lab02_code\matching_source.png"
    img_source = cv2.imread(image_path)
    
    image_path = r"D:\Computer-Vision-2025-Jan-25_19-39-22-456\Computer-Vision-2025-Jan-25_19-39-22-456\viewer\files\lab02 assignment\lab02 assignment\lab02_code\matching_template.png"
    img_template = cv2.imread(image_path)
    if img_source is None:
        raise FileNotFoundError(f"Could not find or open source image at ")
    if img_template is None:
        raise FileNotFoundError(f"Could not find or open template image at ")

    # 2. Convert images to grayscale (recommended for template matching)
    source_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    # 3. Perform template matching
    #    - Methods include: TM_CCOEFF_NORMED, TM_CCORR_NORMED, TM_SQDIFF_NORMED, etc.
    #    - Here we use TM_CCOEFF_NORMED, which works well in many cases.
    result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # 4. Decide on a matching threshold
    #    - Only locations with a matching score >= threshold will be considered valid.
    threshold = 0.8

    # 5. Get the template dimensions (needed to draw bounding boxes)
    tH, tW = template_gray.shape[:2]

    # 6. Find locations in the result image that exceed the threshold
    loc = np.where(result >= threshold)

    # 7. Prepare a list to store bounding boxes
    #    Each bounding box will be a tuple: (x1, y1, x2, y2)
    bounding_boxes = []

    # 8. Draw bounding boxes on the source image
    for pt in zip(*loc[::-1]):
        # pt = (x, y) top-left corner of the match in img_source
        x1, y1 = pt[0], pt[1]
        x2, y2 = x1 + tW, y1 + tH
        bounding_boxes.append((x1, y1, x2, y2))

        # Draw a rectangle around the match on the source image
        cv2.rectangle(
            img_source,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),
            thickness=2
        )

    # 9. Show the final result using either OpenCV or matplotlib
    
    cv2.imshow("Template Matching Result", img_source)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  

    # Return the list of bounding boxes
    return bounding_boxes


if __name__ == "__main__":
    main()

