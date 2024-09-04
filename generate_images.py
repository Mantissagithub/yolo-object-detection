import os 
import cv2 as cv

cap = cv.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("please open the camera")

output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv.imshow('webcam', frame)

    k = cv.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing....")
        break
    elif k%256 == ord('s'):
        img_path = os.path.join(output_dir, "opencv_frame_{}.png".format(img_counter))
        cv.imwrite(img_path, frame)
        print("saved frame {}".format(img_counter))
        img_counter += 1

cap.release()
cv.destroyAllWindows()