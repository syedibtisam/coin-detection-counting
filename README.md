# **Coin Detection and Counting** 🪙

A **computer vision project** that detects and counts coins in images using **Python, OpenCV, and NumPy**.

---

## **Project Overview**
This project implements **image processing and object detection** techniques to identify circular objects (coins) in an image and count them. It utilizes OpenCV’s **Hough Circle Transform** to detect circular shapes and outputs the total number of detected coins.

---

## **Features**
✅ Image preprocessing with **grayscale conversion** and **Gaussian blur**  
✅ **Edge detection** using Canny Edge Detector  
✅ **Circle detection** using Hough Circle Transform  
✅ **Coin counting and visualization** with bounding circles  

---

## **📂 Project Structure**
```
📦 coin-detection-counting
│-- 📂 images/        # Sample images for testing
│-- 📜 main.py        # Main script for coin detection
│-- 📜 testing_hu.py  # Script for testing image moments
│-- 📜 README.md      # Project documentation
```

---

## **Tech Stack**
- **Python**
- **OpenCV** (Computer Vision)
- **NumPy** (Numerical Computing)

---

## **Example Output**
Sample input:  
![Sample Image](images/coins.jpg)

Output (Detected Coins):  
![Detected Coins](images/coin_10.jpg)

---

## **🛠️ Installation & Usage**
### **Install Dependencies**
Ensure you have **Python 3.x** installed. Then, install the required libraries:
```bash
pip install opencv-python numpy
```

### **Run the Script**
```bash
python main.py
```

### **Output**
The script will:
- Load the input image
- Detect and highlight coins in the image
- Print the total number of detected coins

---

## **Example Code Snippet**
```python
import cv2
import numpy as np

# Load image
image = cv2.imread('images/coins.jpg', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# Edge detection
edges = cv2.Canny(blurred, 50, 150)

# Circle detection
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30)

if circles is not None:
    print(f"Detected {len(circles[0])} coins.")
```

---

## **Resources**
- OpenCV Hough Circle Transform: [Documentation](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html)

---

## **License**
This project is **open-source** and free to use under the **MIT License**.

---

## **Contributing**
Feel free to **fork this repo** and submit a pull request if you have improvements!

---

## **Repository Link**
🔗 **GitHub Repo:** [Coin Detection and Counting](https://github.com/syedibtisam/coin-detection-counting)

---


