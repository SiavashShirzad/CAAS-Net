from model import ModelBuilder
from pipeline import DataPipeLine
import tensorflow as tf


data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'

data_pipeline = DataPipeLine(data_path, data_frame, mask_path, 3, 3)
dataset = data_pipeline.dataset_generator()

model_name = 'ResNetRSTridentNet'

model = ModelBuilder(512, 6, model_name)

print(model.summary())

