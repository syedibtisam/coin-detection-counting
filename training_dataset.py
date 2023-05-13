import cv2
import numpy as np
from sklearn import svm, model_selection
from skimage.feature import hog
import os
import glob
import joblib

clf = svm.SVC(decision_function_shape='ovo')


def adaptive_histogram_processing(grayImage):
    # Local Histogram processing to enhance the image locally by dividing into small segments
    contrast_limiting_threshold = 3.0
    ahp = cv2.createCLAHE(clipLimit=contrast_limiting_threshold, tileGridSize=(8,8))
    # Applying to grayscale image
    return ahp.apply(grayImage)


def gaussian_blur(grayImage):
    # gaussian blur, to reduce noise in the image
    controlBlurness = 0
    kernel_size = 15
    blur = cv2.GaussianBlur(grayImage, (kernel_size,kernel_size), controlBlurness)
    return blur


def preprocessed_input_coin_image(path):
    input_image_gray = preprocessing(path)
    input_image_gray = cv2.resize(input_image_gray, (128, 128))
    return input_image_gray

def load_image_in_gray(path):
    try:
        image = cv2.imread(path)
        if image is not None:
            # Convert the image to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print('Error:', e)
    return None


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


def predict_coin(grayImage):
    # Extract HOG features from the preprocessed image
    input_features = hog(grayImage, orientations=11, pixels_per_cell=(12, 12), cells_per_block=(1, 1))
    # input_features = scaler.transform(input_features)
    input_features = input_features.reshape(1, -1)
    predicted_class = clf.predict(input_features)
    return predicted_class


def training_coins_database(path,clf):
    # The path to your coin images, organized into subdirectories by denomination
    # A list to hold the features and labels
    features = []
    labels = []
    # For each denomination directory
    for denomination_dir in glob.glob(os.path.join(data_dir, '*')):
        # Get the denomination from the directory name
        denomination = os.path.basename(denomination_dir)
        # print("Coin ",denomination)
        
        # For each image in the denomination directory
        for img_path in glob.glob(os.path.join(denomination_dir, '*.jpg')) + glob.glob(os.path.join(denomination_dir, '*.jpeg'))+ glob.glob(os.path.join(denomination_dir, '*.png')):
            # Read the image
            img = preprocessed_input_coin_image(img_path)


            # Compute HOG features
            fd = hog(img, orientations=11, pixels_per_cell=(12, 12),
                        cells_per_block=(1, 1))
            # reshaping the feature vector
            # Append the features and label to their respective lists
            # features.append(fd.reshape(-1))  
            features.append(fd)
            labels.append(denomination)

    # Convert the features and labels to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.05, random_state=42)

    # Create and train an SVM
    # clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    # Save the trained model to disk
    joblib.dump(clf, 'svm_model.pkl')
    
# The path to your coin images, organized into subdirectories by denomination
data_dir = 'coins/'
path = "test_coin11.jpeg"
training_coins_database(data_dir,clf)

# preprocessedImage = preprocessed_input_coin_image(path)
# predicted_class = predict_coin(preprocessedImage)
# print(f"Predicted class: {predicted_class}")

