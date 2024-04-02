# detection.py
import cv2
import numpy as np

def initialize_camera_and_model():
    cap = cv2.VideoCapture(0)
    net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
    return cap, net

def detect_faces_and_create_mask(frame, net):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    return detections

def process_frame(frame, detections, mask_enabled):
    mask = np.zeros_like(frame)
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x2, y2) = box.astype("int")
            if mask_enabled:
                mask[y:y2, x:x2] = frame[y:y2, x:x2]
            else:
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    return mask if mask_enabled else frame

def display_frame(window_name, frame):
    cv2.imshow(window_name, frame)

def run_face_detection():
    cap, net = initialize_camera_and_model()
    mask_enabled = False
    window_name = 'Webcam Stream - SSD Face Detection'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_faces_and_create_mask(frame, net)
        processed_frame = process_frame(frame, detections, mask_enabled)
        display_frame(window_name, processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            mask_enabled = not mask_enabled

    cap.release()
    cv2.destroyAllWindows()

