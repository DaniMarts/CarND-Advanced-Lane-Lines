"""
A program to assist with tuning the color, magnitude and gradient threshold filter parameters.
Trackbars are used to visually tune the parameters of the filters using test images.
"""

import cv2
import numpy as np
from glob import glob


def grad_mag(img, sobel_kernel=5):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x and y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8(sobel * 255 / np.max(sobel))

    return sobel

    
def grad_dir(img, sobel_kernel=15):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x and y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.abs(sobel_x)
    abs_sobel_y = np.abs(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # This is the direction in which the color intensity changes the most around each pixel
    directions = np.arctan2(abs_sobel_x, abs_sobel_y)

    return directions


test_images = [cv2.imread(image_name) for image_name in glob("output_images/Undistorted_images/*.jpg")]
# number of images
count = len(test_images)

# for faster processing, let's save the gradient magnitudes and directions for each image
# we can later filter these to obtain a suitable threshold for each
grad_magnitudes = [grad_mag(image) for image in test_images]
grad_directions = [grad_dir(image) for image in test_images]

current_image = 0

# Create a window
cv2.namedWindow('Yellow controls', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED )
cv2.namedWindow('White controls', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED )
cv2.namedWindow('changes', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow("Gradient magnitude threshold", cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow("Gradient direction threshold", cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED)

def render(_):
    image = np.copy(test_images[current_image])
    # get current positions of the trackbars
    YlowH = cv2.getTrackbarPos('Y lowH', 'Yellow controls')
    YhighH = cv2.getTrackbarPos('Y highH', 'Yellow controls')
    YlowS = cv2.getTrackbarPos('Y lowS', 'Yellow controls')
    YhighS = cv2.getTrackbarPos('Y highS', 'Yellow controls')
    YlowV = cv2.getTrackbarPos('Y lowV', 'Yellow controls')
    YhighV = cv2.getTrackbarPos('Y highV', 'Yellow controls')

    WlowH = cv2.getTrackbarPos('W lowH', 'White controls')
    WhighH = cv2.getTrackbarPos('W highH', 'White controls')
    WlowS = cv2.getTrackbarPos('W lowS', 'White controls')
    WhighS = cv2.getTrackbarPos('W highS', 'White controls')
    WlowV = cv2.getTrackbarPos('W lowV', 'White controls')
    WhighV = cv2.getTrackbarPos('W highV', 'White controls')

    mag_low = cv2.getTrackbarPos("Mag low", "Gradient magnitude threshold")
    mag_high = cv2.getTrackbarPos("Mag high", "Gradient magnitude threshold")
    
    dir_low = np.radians(cv2.getTrackbarPos("Dir low", "Gradient direction threshold"))
    dir_high = np.radians(cv2.getTrackbarPos("Dir high", "Gradient direction threshold"))

    # convert color to hsv because it is easy to track colors in this color model
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    Y_lower_hsv = np.array([YlowH, YlowS, YlowV])
    Y_higher_hsv = np.array([YhighH, YhighS, YhighV])

    W_lower_hsv = np.array([WlowH, WlowS, WlowV])
    W_higher_hsv = np.array([WhighH, WhighS, WhighV])

    # mag_thresh = grad_magnitudes[current_image] >= mag_low & grad_magnitudes[current_image] <= mag_high
    
    mag_thresh = cv2.inRange(grad_magnitudes[current_image], mag_low, mag_high)
    dir_thresh = cv2.inRange(grad_directions[current_image], dir_low, dir_high)
    mag_dir_mask = mag_thresh & dir_thresh

    # Apply the cv2.inrange method to create a mask
    color_mask = cv2.inRange(hsv, Y_lower_hsv, Y_higher_hsv) | cv2.inRange(hsv, W_lower_hsv, W_higher_hsv)
    color_mag_dir = np.dstack((np.zeros_like(mag_dir_mask), mag_dir_mask, color_mask))    
    # mask = color_mask | mag_dir_mask
    # Apply the mask on the image to extract the original color
    # frame = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow('changes', frame)
    cv2.imshow("changes", color_mag_dir)

 
# Yellow controls
cv2.createTrackbar('Y lowH','Yellow controls', 17,179,render)
cv2.createTrackbar('Y highH','Yellow controls',25,179,render)
 
cv2.createTrackbar('Y lowS','Yellow controls',95,255,render)
cv2.createTrackbar('Y highS','Yellow controls',255,255,render)
 
cv2.createTrackbar('Y lowV','Yellow controls',160,255,render)
cv2.createTrackbar('Y highV','Yellow controls',255,255,render)
 
# White controls
cv2.createTrackbar('W lowH','White controls',0,179,render)
cv2.createTrackbar('W highH','White controls',80,179,render)
 
cv2.createTrackbar('W lowS','White controls',0,255,render)
cv2.createTrackbar('W highS','White controls',30,255,render)
 
cv2.createTrackbar('W lowV','White controls',200,255,render)
cv2.createTrackbar('W highV','White controls',255,255,render)

# Gradient Magnitude threshold controls
cv2.createTrackbar('Mag low','Gradient magnitude threshold',30,255,render)
cv2.createTrackbar('Mag high','Gradient magnitude threshold',255,255,render)

# Gradient direction threshold controls
cv2.createTrackbar('Dir low','Gradient direction threshold',20,90,render)
cv2.createTrackbar('Dir high','Gradient direction threshold',50,90,render)

render(0)

while True:
    key = cv2.waitKeyEx(0)
    # Press q to exit
    if key == ord('q') or key == ord('Q'):
        break
    # if the key is 4, go back one image
    elif key == ord('4'):
        # the % count is to make sure it stays within the range
        current_image = (current_image + count - 1) % count
        render(0)
    # if the key 6 is pressed, go to the next image
    elif key == ord('6'):
        current_image = (current_image + 1) % count
        render(0)
        
# cap.release()
cv2.destroyAllWindows()
