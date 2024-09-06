from ultralytics import YOLO
import cv2 as cv
import numpy as np

def center(frame, bounding_boxes):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
    ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)
    
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        mask[y1:y2, x1:x2] = thresh[y1:y2, x1:x2]
        contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv.contourArea)
            area = cv.contourArea(largest_contour)
            M = cv.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    cv.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                    cv.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                    cv.putText(frame, "center", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    print(f"Contour Center - x: {cx} y: {cy}")

    cv.imwrite("output_with_boxes.png", frame)

model = YOLO("best (2).pt")
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model.predict(source=frame, conf=0.45)  # Adjust confidence threshold
    bounding_boxes = []
    for result in results:
        for box in result.boxes:
            bounding_boxes.append(box.xyxy[0].tolist())

    center(frame, bounding_boxes)
    cv.imshow("Detection", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        print("Detection stopped.")
        break

cap.release()
cv.destroyAllWindows()
