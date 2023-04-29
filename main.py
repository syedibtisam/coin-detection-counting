import cv2
import numpy as np

def adaptiveThresholding(grayImage):
    thresh = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def adaptiveHistogramProcessing(grayImage):
    # Create an Adaptive Histogram Equalization object
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    # Apply AHE to the grayscale image
    gray_ahe = clahe.apply(grayImage)
    return gray_ahe

def load_Image_In_Gray(path):
    image = cv2.imread(path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def gaussianBlur(grayImage):
    controlBlurness = 0
    kernel_size = 15
    blur = cv2.GaussianBlur(grayImage, (kernel_size,kernel_size), controlBlurness)
    return blur

def cannyEdge(gaussianBlurred):
    lower_threshold = 50 # discard when lower this
    higher_threshold = 150 # strong edges when higher this
    edges = cv2.Canny(gaussianBlurred, lower_threshold, higher_threshold)
    return edges


def houghCircle(blurred_image):
    # circles = cv2.HoughCircles(imageWith_edges, cv2.HOUGH_GRADIENT, 1, 1,
    #                        param1=50, param2=40, minRadius=1, maxRadius=120)
    circles = cv2.HoughCircles(
    blurred_image,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=10, # minimum distance between center of detected circles
    param1=50, # to detect edge, canny edge
    param2=35, # smaller value, results in false circles
    minRadius=10, # min radius of circle to detect
    maxRadius=120, # max radius of circle to detect
    )
    return circles


def preprocessing(img):
    image = load_Image_In_Gray(img)
    # Apply adaptive thresholding
    image = adaptiveHistogramProcessing(image)
    # Apply Gaussian blur to reduce noise
    image = gaussianBlur(image)
    # Apply Canny edge detection
    # image = cannyEdge(image)

    # Apply Hough Circle Transform to detect circles
    circles = houghCircle(image)
    # print(circles)
    return circles


def processing(circles,path):
    image = cv2.imread(path)
    totalCirclesDetected = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # print(circles)
        circles = filtered_circles(circles)

        for (x, y, r) in circles:
            totalCirclesDetected+=1
            # Draw the circle and its center
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 1, (0, 0, 255), 2)

            # Print the detected circle's center and radius
            print(f"Circle center: ({x}, {y}), Radius: {r}")

        # Display the output image
        cv2.imshow("Detected Circles", image)
        cv2.waitKey(0)
        print("total Circles Detected:",totalCirclesDetected)
    else:
        print("No circles detected.")


def circles_overlap(c1, c2, overlap_threshold):
    (x1, y1, r1) = c1
    (x2, y2, r2) = c2
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance < (r1 + r2) * overlap_threshold


def filtered_circles(circles):
    overlap_threshold = 0.8  # less than 1 means that circles can overlap
    filtered_circles = []
    for (x, y, r) in circles:
        overlap = False
        for (x2, y2, r2) in filtered_circles:
            if circles_overlap((x, y, r), (x2, y2, r2), overlap_threshold):
                overlap = True
                break
        if not overlap:
            filtered_circles.append((x, y, r))
    return filtered_circles

def main():
    path = "pakistani_coins.jpeg"
    path = "coins.jpg"
    path = "coins3.jpeg"
    circles = preprocessing(path)
    processing(circles,path)

def display_image(imageDisplayName,img):
    cv2.imshow(imageDisplayName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()