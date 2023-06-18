import cv2
import dlib

video_capture = cv2.VideoCapture(0)

while True:
    rectangle, frame = video_capture.read()

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector_hog = dlib.get_frontal_face_detector()
    detectors = face_detector_hog(image_gray, 1)

    # Draw a rectangle Box around the faces
    for face in detectors:
        l,t,r,b=face.left(),face.top(),face.right(),face.bottom()
        cv2.rectangle(frame, (l, t), (r, b), (255, 255, 0), 2)

    cv2.imshow('WebCam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyALLWindows()