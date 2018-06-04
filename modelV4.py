import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


def readCSV(csvfilePath, csvfileName):
    lines = []
    with open(csvfilePath+csvfileName) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        return lines #A line is a four-element list

def flipImg(img, measurement):
    """
    Flip an image vertically
    :param img: 
    :param measurement: 
    :return: img, measurement
    """
    newImg = cv2.flip(img,1)
    newMeasurement = measurement*(-1.0)
    return newImg, newMeasurement




def loadImageAndMeasurement(currentImgPath, correction=0.2):
    images = []
    measurements = []
    counter = 0
    lines = readCSV(csvfilePath=csvfilePath, csvfileName=csvfileName)
    for line in lines:
        centreImgFileName = line[0].split('\\')[-1]
        centreImg = cv2.cvtColor(cv2.imread(currentImgPath + centreImgFileName), cv2.COLOR_BGR2RGB)
        images.append(centreImg)
        measurement = float(line[3])
        measurements.append(measurement)

        flip_img, flip_measurement = flipImg(centreImg, measurement)
        images.append(flip_img)
        measurements.append(flip_measurement)

        leftImgFileName = line[1].split('\\')[-1]
        leftImg = cv2.cvtColor(cv2.imread(currentImgPath + leftImgFileName), cv2.COLOR_BGR2RGB)

        images.append(leftImg)
        measurements.append(measurement+correction)

        flip_img, flip_measurement = flipImg(leftImg, measurement+correction)
        images.append(flip_img)
        measurements.append(flip_measurement)

        rightImgFileName = line[2].split('\\')[-1]
        rightImg = cv2.cvtColor(cv2.imread(currentImgPath + rightImgFileName), cv2.COLOR_BGR2RGB)
        images.append(rightImg)
        measurements.append(measurement-correction)

        flip_img, flip_measurement = flipImg(rightImg, measurement-correction)
        images.append(flip_img)
        measurements.append(flip_measurement)

    assert (len(images) == len(measurements))


    return np.array(images), np.array(measurements)



def buildModel(inputShape):
    """
    Create modified version of the nVideo model (with cropping)
    :param inputShape:  
    :return: model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=cropping))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

def train_model(model, inputDate, outputDate, modelFilePath, modelFileName, epochs):
    model.compile(loss='mse', optimizer='adam')
    model.fit(inputDate, outputDate, validation_split=0.2, shuffle = True, nb_epoch = epochs)
    model.save(modelFilePath+modelFileName)
    print("Model saved as ", modelFilePath+modelFileName)






csvfilePath = './data/'
csvfileName = 'driving_log.csv'
currentImgPath = './data/IMG/'
cropping = ((50, 20), (0, 0))
inputShape = (160,320,3)
modelFilePath = './models/'
modelFileName = 'model.h5'
epochs = 3


X_train, y_train = loadImageAndMeasurement(currentImgPath=currentImgPath)
model = buildModel(inputShape=inputShape)
train_model(model, X_train, y_train, modelFilePath, modelFileName, epochs)






