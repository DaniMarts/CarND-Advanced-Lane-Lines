import cv2
import numpy as np
 
def nothing(x):
    pass

# Create a window
cv2.namedWindow('Yellow controls', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED )
cv2.namedWindow('White controls', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED )

cv2.namedWindow('changes', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED)
 
# Yellow controls
# create trackbars for Yellow change
cv2.createTrackbar('Y lowH','Yellow controls',0,179,nothing)
cv2.createTrackbar('Y highH','Yellow controls',179,179,nothing)
 
cv2.createTrackbar('Y lowS','Yellow controls',0,255,nothing)
cv2.createTrackbar('Y highS','Yellow controls',255,255,nothing)
 
cv2.createTrackbar('Y lowV','Yellow controls',0,255,nothing)
cv2.createTrackbar('Y highV','Yellow controls',255,255,nothing)
 
# White controls
cv2.createTrackbar('W lowH','White controls',0,179,nothing)
cv2.createTrackbar('W highH','White controls',179,179,nothing)
 
cv2.createTrackbar('W lowS','White controls',0,255,nothing)
cv2.createTrackbar('W highS','White controls',255,255,nothing)
 
cv2.createTrackbar('W lowV','White controls',0,255,nothing)
cv2.createTrackbar('W highV','White controls',255,255,nothing)

while(True):
    # frame=np.copy(FRAME)
    frame = cv2.imread("test_images/test6.jpg")

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
    
    # convert color to hsv because it is easy to track colors in this color model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Y_lower_hsv = np.array([YlowH, YlowS, YlowV])
    Y_higher_hsv = np.array([YhighH, YhighS, YhighV])

    W_lower_hsv = np.array([WlowH, WlowS, WlowV])
    W_higher_hsv = np.array([WhighH, WhighS, WhighV])

    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, Y_lower_hsv, Y_higher_hsv) | cv2.inRange(hsv, W_lower_hsv, W_higher_hsv)
    # Apply the mask on the image to extract the original color
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('changes', frame)
    # Press q to exit

    # key = cv2.waitKey(0)
    # print(key)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()