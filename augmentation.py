import cv2
import tensorflow as tf
import numpy as np


class AugmentationBuilder:
    def __init__(self, zoom, rotation, contrast, brightness):
        self.zoom = zoom
        self.rotation = rotation
        self.contrast = contrast
        self.brightness = brightness

    def image_augmentation(self):
        pass

    def mask_augmentation(self):
        pass

    def augmentation(self, image, mask):
        pass
