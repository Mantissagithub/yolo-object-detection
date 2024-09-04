from ultralytics import YOLO
import cv2

model = YOLO("best (2).pt")

results = model.predict(source="0", show=True, conf=0.56)

while True:
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

    elif key == 27: 
        print("Detection stopped.")
        break

cv2.destroyAllWindows()