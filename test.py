from ultralytics import YOLO
import cv2 as cv
import numpy as np

def center(frame):
    frame = cv.imread(frame)
    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
    
    # Apply thresholding
    ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)
    cv.imwrite("thresh.png", thresh)
    
    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(thresh.shape[:2], dtype='uint8')
    cv.drawContours(blank, contours, -1, (255, 0, 0), 1)
    cv.imwrite("Contours.png", blank)

    # Find the center of each contour
    for i in contours:
        M = cv.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv.drawContours(frame, [i], -1, (0, 255, 0), 2)
            cv.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv.putText(frame, "center", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f"x: {cx} y: {cy}")
    cv.imwrite("image.png", frame)


center('image.png')