from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.imageTools import ImageToArrayPreprocessor
from arc_1 import IncludeNet
from utils.dataLoader import SimpleDatasetLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from keras.optimizers import Adam
from imutils import paths
from utils.preprocessor import preprocessor
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="dataset")
ap.add_argument("-m", "--model", required=True, help="path to save model")
args = vars(ap.parse_args())
size = 50
ep = 500
dpt = 3

print("[INFO] loading Images")
imagePaths = list(paths.list_images(args["dataset"]))
sp = preprocessor(size, size)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
print(labels)
data = data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt =Adam(lr = 0.25)
model = IncludeNet.build(width=size, height=size, depth=dpt, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=size, epochs=ep, verbose=1)
print("Saving network")
model.save('model6.hdf5')
print("Network have been saved")

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=size)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=["covid", "normal"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, ep), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, ep), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, ep), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, ep), H.history["val_accuracy"], label="val_acc")
plt.title("results")
plt.xlabel("Epoch #")
plt.ylabel("Loss/ACC")
plt.legend()
plt.show()
