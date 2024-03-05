
import os

from keras.models import Sequential
from imutils import paths
import random, shutil

from keras.layers import BatchNormalization
from keras.layers.convolutional import SeparableConv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K

import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

# -------------------- Global Variables -----------------------
INPUT_DATASET = "datasets/original"

BASE_PATH = "datasets/idc"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


NUM_EPOCHS=40
INIT_LR=1e-2
BS=32
#---------------------- Build Dataset ------------------------
# originalPaths=list(paths.list_images(INPUT_DATASET))
# random.seed(7)
# random.shuffle(originalPaths)


# index=(int(len(originalPaths)*TRAIN_SPLIT))
# trainPaths = originalPaths[:index]
# testPaths = originalPaths[index:]

# index = int(len(trainPaths)*VAL_SPLIT)
# valPaths = trainPaths[:index]
# trainPaths = trainPaths[index:]

# datasets = [("training", trainPaths, TRAIN_PATH),
#             ("validation", valPaths, VAL_PATH),
#             ("testing", testPaths, TEST_PATH)]
# for (setType, originalPaths, basePath) in datasets:
#     print(f'Building {setType} set')

#     if not os.path.exists(basePath):
#         print(f'Building directory {basePath}')
#         os.makedirs(basePath)
    
#     for path in originalPaths:
#         file = path.split(os.path.sep)[-1]
#         label = file[-5:-4]

#         labelPath=os.path.sep.join([basePath,label])
#         if not os.path.exists(labelPath):
#                 print(f'Building directory {labelPath}')
#                 os.makedirs(labelPath)
#         newPath=os.path.sep.join([labelPath, file])
#         shutil.copy2(path, newPath)
# -------------------- Network (CNN) ---------------------------------------
class CancerNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        shape=(height, width, depth)
        channelDim=-1

        if K.image_data_format()=="Channels_first":
            shape=(depth, height, width)
            channelDim=1

        model.add(SeparableConv2D(32, (3,3), padding="same",input_shape=shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(SeparableConv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(SeparableConv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

# ------------- Declaring data loaders and their augmentations ----------------------

# import pdb; pdb.set_trace()
trainPaths=list(paths.list_images(TRAIN_PATH))
lenTrain=len(trainPaths)
lenVal=len(list(paths.list_images(VAL_PATH)))
lenTest=len(list(paths.list_images(TEST_PATH)))

import pdb; pdb.set_trace()
trainLabels=[int(p.split(os.path.sep)[-2]) if (p.split(os.path.sep)[-2] in ['0','1']) else int(-2)  for p in trainPaths]
idx=np.where((np.array(trainLabels)==-2))
# trainLabels.pop(idx[0][0])
# trainPaths.pop(idx[0][0])
trainLabels=np_utils.to_categorical(trainLabels)
classTotals=trainLabels.sum(axis=0)
classWeight=classTotals.max()/classTotals

trainAug = ImageDataGenerator(
  rescale=1/255.0,
  rotation_range=20,
  zoom_range=0.05,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.05,
  horizontal_flip=True,
  vertical_flip=True,
  fill_mode="nearest")

valAug=ImageDataGenerator(rescale=1 / 255.0)

trainGen = trainAug.flow_from_directory(TRAIN_PATH, class_mode="categorical", target_size=(50,50), color_mode="rgb", shuffle=True, batch_size=BS)

valGen = valAug.flow_from_directory(VAL_PATH, class_mode="categorical", target_size=(50,50), color_mode="rgb", shuffle=False, batch_size=BS)

testGen = valAug.flow_from_directory(TEST_PATH, class_mode="categorical", target_size=(48,48), color_mode="rgb", shuffle=False, batch_size=BS)

# create the model and train it on data
model=CancerNet.build(width=50,height=50,depth=3,classes=3)
opt=Adagrad(lr=INIT_LR,decay=INIT_LR/NUM_EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

import pdb; pdb.set_trace()
M=model.fit_generator(trainGen, steps_per_epoch=lenTrain//BS, validation_data=valGen, validation_steps=lenVal//BS, class_weight=classWeight, epochs=NUM_EPOCHS)

print("Now evaluating the model")
testGen.reset()
pred_indices=model.predict_generator(testGen,steps=(lenTest//BS)+1)

pred_indices=np.argmax(pred_indices,axis=1)

# -------------- Report the results -----------------------

print(classification_report(testGen.classes, pred_indices, target_names=testGen.class_indices.keys()))

cm=confusion_matrix(testGen.classes,pred_indices)
total=sum(sum(cm))
accuracy=(cm[0,0]+cm[1,1])/total
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print(cm)
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')

N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), M.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), M.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on the IDC Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')
        