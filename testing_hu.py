import cv2
import numpy as np

# Load the images of the extracted coin and the reference coin
coin1 = cv2.imread('coin_1.jpg', cv2.IMREAD_GRAYSCALE)
coin2 = cv2.imread('coin_1.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the Hu moments of the extracted coin
moments1 = cv2.moments(coin1)
huMoments1 = cv2.HuMoments(moments1)

# Calculate the Hu moments of the reference coin
moments2 = cv2.moments(coin2)
huMoments2 = cv2.HuMoments(moments2)

# Calculate the Euclidean distance between the Hu moments
distance = np.sqrt(np.sum((huMoments1 - huMoments2) ** 2))

# Print the distance
print(distance)
