import numpy as np
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


class MultipleClassSegmentationMetrics:
    def __init__(self, channels):
        self.channels = channels

    def dice_multi_coef(self, y_true, y_pred, smooth=1e-7):
        y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=self.channels)[..., 1:])
        y_pred_f = K.flatten(y_pred[..., 1:])
        intersect = K.sum(y_true_f * y_pred_f, axis=-1)
        denom = K.sum(y_true_f + y_pred_f, axis=-1)
        return K.mean((2. * intersect / (denom + smooth)))
