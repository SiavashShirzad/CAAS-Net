from models import BuildDeepLabV3, BuildEfficientB0Unet
builder = BuildEfficientB0Unet()
model = builder(512,5)
print(model.summary())