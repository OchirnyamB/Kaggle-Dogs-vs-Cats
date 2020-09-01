from model.alexnet import AlexNet 
from keras.applications import imagenet_utils
from preprocessing.simplepreprocessor import SimplePreprocessor
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-weights", "--weights", required=True, help="path to the pre-trained weights to use")
args = vars(ap.parse_args()) 

classLabels = ["cat", "dog"]

print("[INFO] loading and pre-processing image...")
sp = SimplePreprocessor(227, 227)
image = cv2.imread(args["image"])
image = sp.preprocess(image)
image = img_to_array(image)

# Image is now represented as a Numpy array of shape (1, inputShape[0], inputShape[1], 3)
image = np.expand_dims(image, axis=0)

print("[INFO] loading pre-trained AlexNet model...")
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.load_weights(args["weights"])

print("[INFO] classifying an image...")
preds = model.predict(image)
print(preds)
preds = preds.argmax(axis=1)
print(preds)
# Load the image via OpenCV, draw the top prediction on the image, and diplsy the image to our screen
orig = cv2.imread(args["image"])
cv2.putText(orig, "Label : {}".format(classLabels[preds[0]]), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)