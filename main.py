import cv2
import numpy as np

def adaptive_thresholding(grayImage):
    # Adaptive thresholding in order to present better results
    maximum_value_to_assign = 255
    local_neighbourhood_window = 9
    adjust_contrast = 3 # to subtract from mean value 
    return cv2.adaptiveThreshold(grayImage, 
                                  maximum_value_to_assign, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY_INV, 
                                  local_neighbourhood_window, 
                                  adjust_contrast)

def adaptive_histogram_processing(grayImage):
    # Local Histogram processing to enhance the image locally by dividing into small segments
    contrast_limiting_threshold = 3.0
    ahp = cv2.createCLAHE(clipLimit=contrast_limiting_threshold, tileGridSize=(8,8))
    # Applying to grayscale image
    return ahp.apply(grayImage)

def load_image_in_gray(path):
    try:
        image = cv2.imread(path)
        if image is not None:
            # Convert the image to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print('Error:', e)
    return None
    

def gaussian_blur(grayImage):
    # gaussian blur, to reduce noise in the image
    controlBlurness = 0
    kernel_size = 15
    blur = cv2.GaussianBlur(grayImage, (kernel_size,kernel_size), controlBlurness)
    return blur

def canny_edge(gaussianBlurred):
    lower_threshold = 30 # discard when lower this
    higher_threshold = 130 # strong edges when higher this
    edges = cv2.Canny(gaussianBlurred, lower_threshold, higher_threshold)
    return edges


def hough_circle(blurred_image):
    return cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10, # minimum distance between center of detected circles
        param1=50, # to detect edge, canny edge
        param2=35, # smaller value, results in false circles
        minRadius=10, # min radius of circle to detect
        maxRadius=120, # max radius of circle to detect
    )


def preprocessing(img):
    try:
        image = load_image_in_gray(img)
        if image is not None:
            # Apply adaptive thresholding
            image = adaptive_histogram_processing(image)
            # Apply Gaussian blur to reduce noise
            image = gaussian_blur(image)
            # Apply Canny edge detection
            # image = cannyEdge(image)
            return image
        else:
            return None
    except Exception as e:
        print('Error:', e)

    return None


def calculate_coins_hu_moments():
    coins_hu={}
    coins = [1,2,5,10]
    for i in range(4):
        if coins[i] == 1:
            local = cv2.imread(f"coin_{coins[i]}.jpeg", cv2.IMREAD_GRAYSCALE)
        else:
            local = cv2.imread(f"coin_{coins[i]}.jpg", cv2.IMREAD_GRAYSCALE)
        coins_hu[coins[i]] = calculate_hu_moments(local)

    return coins_hu


def detecting_circles(preprocessedImage,path):
    # Used Hough transform function to find out all the circles in the image
    try:
        circles = hough_circle(preprocessedImage)
        image = cv2.imread(path)
        totalCirclesDetected = 0
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int") # rounding to nearest integer and converting to int
            circles = filtered_circles(circles)

            for (x, y, r) in circles:
                totalCirclesDetected+=1
                # Draw the circle and its center
                cv2.circle(image, (x, y), r, (0, 0, 255), 2)
                cv2.circle(image, (x, y), 1, (255, 0, 0), 2)

            # Display the output image
            cv2.imshow("Detected Circles", image)
            cv2.waitKey(0)
            print("total Circles Detected:",totalCirclesDetected)
        else:
            print("No circles detected.")
        # print(circles)
        
        return circles
    except Exception as e:
        print('Error:', e)
        return "Error"

def circles_overlap(circle1, circle2, overlap_threshold):
    (x1, y1, r1) = circle1
    (x2, y2, r2) = circle2
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


def extract_coin_shape(gray_image, center, radius): # input blurred image from preprocessing
    x, y = center
    r = int(radius * 1.1)
    return gray_image[y - r : y + r, x - r : x + r]


def coins_matching(circles,preprocessedImage):
    coins_hu_moments = calculate_coins_hu_moments()

    total_amount = 0
    for (x, y, r) in circles:
        coin_shape = extract_coin_shape(preprocessedImage, (x, y), r)
        display_image("coin",coin_shape)
        coin_moments = calculate_hu_moments(coin_shape)
        print("-----------")
        for value, template_hu_moments in coins_hu_moments.items():
            distance = compute_distance_btw_moments(coin_moments,template_hu_moments)
            if value == 10:
                print("Coin:",value,'{:.20f}'.format(distance))
            else:
                print("Coin:",str(value)+" ",'{:.20f}'.format(distance))



def calculate_hu_moments(grayscale_image):
    # Calculate the Hu moments of the extracted coin
    moments = cv2.moments(grayscale_image)
    return cv2.HuMoments(moments)

def compute_distance_btw_moments(m1,m2):
    # Euclidean distance between the Hu moments
    return np.sqrt(np.sum((m1 - m2) ** 2))


def main():
    path = "pakistani_coins.jpeg"
    path = "coins.jpg"
    path = "coins3.jpeg"
    path = "coins2.jpg"

    # preprocessing
    preprocessedImage = preprocessing(path)
    if preprocessedImage is not None:
        # detecting the circles
        circles = detecting_circles(preprocessedImage,path)
        if circles == "Error":
            print("error occurred")
        elif circles is None:
            print("No circles detected")
        else:
            # coins matching
            image = load_image_in_gray(path)
            # Apply adaptive thresholding
            image = adaptive_histogram_processing(image)
            coins_matching(circles,image)
    else:
        print("failed to preprocess image")

def display_image(imageDisplayName,img):
    cv2.imshow(imageDisplayName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()