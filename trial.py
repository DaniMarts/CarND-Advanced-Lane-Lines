import cv2

cv2.namedWindow("nothing")

while True:
    k = cv2.waitKeyEx(0)
    if k != -1:
        print(k)
    elif k == ord('q'):
        break

cv2.destroyAllWindows()