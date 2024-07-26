import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam

def load_dataset():
    "loads dataset from s3 bucket"
    trainx_dataset = pd.read_parquet(
        os.path.join(os.environ["SM_CHANNEL_TRAINX"], "trainX.parquet"))
    
    trainy_dataset = pd.read_parquet(
        os.path.join(os.environ["SM_CHANNEL_TRAINY"], "trainY.parquet"))
    
    testx_dataset = pd.read_parquet(
        os.path.join(os.environ["SM_CHANNEL_TESTX"], "testX.parquet"))
    
    testy_dataset = pd.read_parquet(
        os.path.join(os.environ["SM_CHANNEL_TESTY"], "testY.parquet"))
    
    # Convert to NumPy arrays and reshape
    trainx = trainx_dataset.to_numpy().reshape(-1, 64, 64, 3)
    trainy = trainy_dataset['label'].to_numpy()
    testx = testx_dataset.to_numpy().reshape(-1, 64, 64, 3)
    testy = testy_dataset['label'].to_numpy()
    
    return trainx, trainy, testx, testy

def train(args):
    "trains the model"
    trainX, trainY, testX, testY = load_dataset()
    print(trainX.shape, trainY.shape)
    
    #converting label values to onehot encoding
    trainY_onehot = to_categorical(trainY, num_classes=9)
    testY_onehot = to_categorical(testY, num_classes=9)
    

    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    print(vgg16_base.summary() )
    vgg16_base.trainable = False

    #Unfreeze  block4_conv1 layer
    trainableFlag = False
    for layer in vgg16_base.layers:
        if layer.name == 'block4_conv1':
            trainableFlag = True
        layer.trainable = trainableFlag

    #Add a new fully connected layer. 
    vgg16_model = Sequential()
    vgg16_model.add(vgg16_base)
    vgg16_model.add(Flatten())
    vgg16_model.add(Dense(256, activation='relu'))
    vgg16_model.add(Dense(9, activation='softmax'))

    vgg16_model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    history = vgg16_model.fit(trainX, trainY_onehot, epochs=args.epochs, batch_size=args.per_device_train_batch_size, validation_data=       (testX, testY_onehot))
    vgg16_accuracy = vgg16_model.evaluate(testX, testY_onehot, batch_size=args.per_device_eval_batch_size)

    # Save the model in TensorFlow SavedModel format
    vgg16_model.save(os.path.join(args.model_dir, '1'), save_format='tf')
    
    
if __name__ == "__main__":

    # SageMaker passes hyperparameters  as command-line arguments to the script
    # Parsing them below...
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=10)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args, _ = parser.parse_known_args()

    train(args)
