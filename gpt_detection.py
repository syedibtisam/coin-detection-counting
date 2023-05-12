import cv2
import numpy as np
from sklearn import svm, model_selection
from skimage.feature import hog
from skimage import color
import os
import glob

# The path to your coin images, organized into subdirectories by denomination
data_dir = 'coins/'

# A list to hold the features and labels
features = []
labels = []

# For each denomination directory
for denomination_dir in glob.glob(os.path.join(data_dir, '*')):
    # Get the denomination from the directory name
    denomination = os.path.basename(denomination_dir)
    # print("Coin ",denomination)
    
    # For each image in the denomination directory
    for img_path in glob.glob(os.path.join(denomination_dir, '*.jpg')) + glob.glob(os.path.join(denomination_dir, '*.jpeg')):
        # Read the image
        # print(img_path)
        img = cv2.imread(img_path, 0)

        # Resize the image to a standard size (let's say 64x64)
        img = cv2.resize(img, (256, 256))

        # Compute HOG features
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1))

        # Append the features and label to their respective lists
        features.append(fd)
        labels.append(denomination)

# Convert the features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.05, random_state=42)

# Create and train an SVM
clf = svm.SVC()
clf.fit(X_train, y_train)

# Calculate the accuracy of the model on the test set
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
