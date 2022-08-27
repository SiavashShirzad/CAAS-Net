import pydicom as dicom
import os
import numpy as np
import matplotlib
from tqdm import tqdm
import nibabel as nib


def vid_to_array(path):
    vid = dicom.dcmread(path)
    vid = vid.pixel_array


class DataCleaning():
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.affine = None

    def dicom_to_png(self):
        angio_list = os.listdir(self.data_path)
        for name in tqdm(angio_list):
            try:
                for image in os.listdir(self.data_path + "/" + name + "/SR 1"):
                    if image[0] == 'I':
                        i = 0
                        for img in vid_to_array(self.data_path + "/" + name + "/SR 1" + "/" + image):
                            i = i + 1
                            matplotlib.image.imsave(self.save_path + name + "$" + image + "$" + str(i) + ".png", img,
                                                    cmap="gray")
            except:
                for image in os.listdir(self.data_path + "/" + name + "/d0"):
                    if image[0] == 'S':
                        for img in vid_to_array(self.data_path + "/" + name + "/SR 1" + "/" + image):
                            i = i + 1
                            matplotlib.image.imsave(self.save_path + name + "$" + image + "$" + str(i) + ".png", img,
                                                    cmap="gray")
            finally:
                continue

    # recommended method for cleaning data 1. easier to anonymize (smaller header) 2. gives 3d arrays 3. can be used in rilcontour (great free annotation app) 4. preseves the pixel data

    def dicom_to_nifti(self):
        angio_list = os.listdir(self.data_path)
        for cd in tqdm(angio_list[9:]):
            for name in os.listdir(self.data_path + '/' + cd):
                if name[0] == 'd':
                    for image in os.listdir(self.data_path + "/" + cd + '/' + name):
                        try:
                            if image[0] == 'S':
                                nib.save(nib.Nifti1Image(
                                    vid_to_array(self.data_path + "/" + cd + '/' + name + "/" + image),
                                    self.affine),
                                    self.save_path + cd + '$' + name + "$" + image + '.nii.gz')
                        except:
                            continue
