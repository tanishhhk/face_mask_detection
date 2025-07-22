import cv2
import time
import os
import csv
import argparse
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import datetime
import winsound  # For audio alert on Windows


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("--log", action="store_true", help="log detections to CSV")
ap.add_argument("--sound", action="store_true", help="play sound on No Mask detection")
args = vars(ap.parse_args())

# Load models
print("[INFO] loading models...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(args["model"])

# Logging setup
if args["log"]:
    log_file = open("detections_log.csv", mode='w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Timestamp", "Label", "Confidence"])

# Start video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps_start = time.time()
fps_count = 0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        confidence = max(mask, withoutMask) * 100
        label_text = f"{label}: {confidence:.2f}%"

        cv2.putText(frame, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Optional logging
        if args["log"]:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_writer.writerow([timestamp, label, f"{confidence:.2f}%"])

        # Optional audio alert
        if args["sound"] and label == "No Mask":
            winsound.Beep(1000, 300)  # frequency, duration (ms)

    # FPS Counter
    fps_count += 1
    elapsed = time.time() - fps_start
    if elapsed > 1:
        fps = fps_count / elapsed
        fps_text = f"FPS: {fps:.2f}"
        fps_start = time.time()
        fps_count = 0
    else:
        fps_text = ""

    if fps_text:
        cv2.putText(frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, "Press 'q' to quit", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ✅ FIXED: Now these are indented properly inside the loop
    cv2.namedWindow("Face Mask Detector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Face Mask Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Face Mask Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or cv2.getWindowProperty("Face Mask Detector", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
if args["log"]:
    log_file.close()
vs.stop()
cv2.destroyAllWindows()
