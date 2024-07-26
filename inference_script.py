import argparse
import os
# import numpy as np
# import pandas as pd
import tensorflow as tf
from keras.models import load_model


def model_fn(model_dir):
    """
    Loads the trained model from the specified directory.
    Args:
        model_dir (str): Path to the directory containing the saved model.
    Returns:
        model (tf.keras.Model): Loaded Keras model.
    """
    model = tf.keras.models.load_model(os.path.join(model_dir, '1'))
    return model
