from models import BuildDeepLabV3, BuildEfficientB0Unet


class ModelBuilder:

    def __init__(self, image_size, number_classes, model_name):
        self.image_size = image_size
        self.number_classes = number_classes
        self.model_name = model_name

        if self.model_name == "deeplab":
            model_builder = BuildDeepLabV3()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "efficientb0":
            model_builder = BuildEfficientB0Unet()
            self.model = model_builder(self.image_size, self.number_classes)

    def summary(self):
        return self.model.summary()

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, dataset, validation_data, epochs, callbacks):
        return self.model.fit(dataset, validation_data=validation_data, epochs=epochs, callbacks=callbacks)

    def evaluate(self, data):
        return self.model.evaluate(data)

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, data):
        return self.model.evaluate(data)
