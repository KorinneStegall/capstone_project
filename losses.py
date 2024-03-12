# 3 custom loss functions GitHub (Link: https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py)
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import os 
from skimage import io
from tensorflow.keras import backend as K


def tversky(y_true, y_pred, smooth = 1e-6):
    y_true_pos = K.cast(K.flatten(y_true), 'float32')
    y_pred_pos = K.cast(K.flatten(y_pred), 'float32')
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)