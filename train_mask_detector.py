"""
Face Mask Detection System - Training Script
Author: Tanishk Jain
Description: Train a face mask classifier using MobileNetV2 and save the model.
"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import pickle

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="Path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot", help="Base name for plot file (timestamp added automatically)")
ap.add_argument("-e", "--epochs", type=int, default=20, help="Number of training epochs")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
ap.add_argument("-l", "--learning-rate", type=float, default=1e-4, help="Initial learning rate")
args = vars(ap.parse_args())

# Load and preprocess dataset
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# Binarize and one-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Save label binarizer
with open("label_binarizer.pickle", "wb") as f:
    f.write(pickle.dumps(lb))

# Split dataset
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load MobileNetV2 base model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# Build head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Final model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=args["learning_rate"])
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(args["model"], monitor="val_accuracy", save_best_only=True, verbose=1)

# Train model
print("[INFO] Training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=args["batch_size"]),
    steps_per_epoch=len(trainX) // args["batch_size"],
    validation_data=(testX, testY),
    validation_steps=len(testX) // args["batch_size"],
    epochs=args["epochs"],
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate
print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=args["batch_size"])
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Save training plot
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
plot_file = f"{args['plot']}_{timestamp}.png"
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_file)
print(f"[INFO] Training plot saved as {plot_file}")
