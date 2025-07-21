# Face Mask Detection 

A real-time face mask detection system using deep learning and computer vision. This project identifies whether a person is wearing a face mask via webcam input.

> Created and maintained by **Tanishk Jain**.

---

## 🚀 Features

- Real-time video feed processing
- Face detection using OpenCV DNN
- Face mask classification using a pre-trained Keras model
- Color-coded bounding boxes and label confidence
- Lightweight and easy to deploy

---

## 🛠️ Technologies Used

- Python 3.8+
- OpenCV
- TensorFlow / Keras
- NumPy
- imutils

---


## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/tanishhhk/face_mask_detection
cd face-mask-detection
Step 2: Set Up Virtual Environment (Optional but Recommended)

py -3.8 -m venv venv38
venv38\Scripts\activate
Step 3: Install Dependencies
Make sure pip is up-to-date:


python -m pip install --upgrade pip
Then install the requirements:


pip install -r requirements.txt


python detect_mask_video.py
Press q or close the window to stop the video stream.

🧠 Model Info
Face Detector: Based on OpenCV’s pretrained Caffe model.

Mask Classifier: Trained on a custom dataset using MobileNetV2.



📁 Directory Structure

face-mask-detection/
│
├── face_detector/                # Caffe model for face detection
├── dataset/                      # Dataset used for training
├── detect_mask_video.py         # Main video stream detection script
├── train_mask_detector.py       # Script to train the classifier
├── mask_detector.model          # Saved trained model
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
🧪 Future Improvements
Add face mask color and type classification

Export model to TensorFlow Lite for mobile usage

Improve detection speed and accuracy

Web-based interface using Flask or Streamlit

📬 Contact
If you'd like to connect, suggest improvements, or collaborate:

Tanishk Jain
📧 [tanishkjain3011@gmail.com]
📌 [Noida , Uttar pradesh]

📜 License
This project is open-source. You may modify or reuse the code as needed. Attribution is appreciated but not required.


---
