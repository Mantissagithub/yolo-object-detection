import os 
import cv2 as cv
import supervision as sv
from ultralytics import YOLO

# Make sure the model path is correct
model_path = 'best (2).pt'  # Update this to the correct path if necessary

try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Please open the camera")
    exit()

img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotate_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotate_image = label_annotator.annotate(scene=annotate_image, detections=detections)

    cv.imshow('webcam', annotate_image)  # Show the annotated image

    k = cv.waitKey(1)

    if k % 256 == 27:  # Escape key
        print("Escape hit, closing....")
        break

cap.release()
cv.destroyAllWindows()