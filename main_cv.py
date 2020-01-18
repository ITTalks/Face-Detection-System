import cv2
import sys

cur_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

# Specify the path to the hml file, 
# Which is responsible for marking and building the frontal cascade

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    
    # Capture video frame by frame

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE

    )
# Draw rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Real time image output
    cv2.imshow('VIDEO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done release the capture

video_capture.release()
cv2.destroyAllWindows()
