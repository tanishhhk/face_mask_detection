"""
Face Mask Detection System - Image Inference
Author: Tanishk Jain
Updated by: ChatGPT (2025)
Description: Detects face masks in a static image using OpenCV and TensorFlow.
"""

import os
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def detect_mask_on_image(image_path, face_model_dir, mask_model_path, confidence_threshold, save_output=False):
    # Load face detector model
    print("[INFO] Loading face detector model...")
    prototxt_path = os.path.join(face_model_dir, "deploy.prototxt")
    weights_path = os.path.join(face_model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        raise FileNotFoundError("Face detector model files not found in path: " + face_model_dir)

    net = cv2.dnn.readNet(prototxt_path, weights_path)

    # Load face mask classifier model
    print("[INFO] Loading mask detector model...")
    model = load_model(mask_model_path)

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    print("[INFO] Detecting faces and predicting masks...")
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face, verbose=0)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            prob = max(mask, withoutMask) * 100
            label_text = f"{label}: {prob:.2f}%"

            cv2.putText(image, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # Show result
    cv2.imshow("Face Mask Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result image
    if save_output:
        output_path = os.path.splitext(image_path)[0] + "_output.jpg"
        cv2.imwrite(output_path, image)
        print(f"[INFO] Output image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect face masks in a static image.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    parser.add_argument("-f", "--face", default="face_detector",
                        help="Path to face detector model directory (default: face_detector/)")
    parser.add_argument("-m", "--model", default="mask_detector.model",
                        help="Path to trained face mask detector model (default: mask_detector.model)")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="Minimum probability to filter weak face detections.")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save output image with annotations.")
    
    args = parser.parse_args()
    detect_mask_on_image(args.image, args.face, args.model, args.confidence, args.save)
