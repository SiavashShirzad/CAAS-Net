from models import DeepLabV3Builder, EfficientB0UnetBuilder, VGG16ModelBuilder


class ModelBuilder:

    def __init__(self, image_size, number_classes, model_name):
        self.image_size = image_size
        self.number_classes = number_classes
        self.model_name = model_name

        if self.model_name == "deeplab":
            model_builder = DeepLabV3Builder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "efficientb0":
            model_builder = EfficientB0UnetBuilder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "vgg16":
            model_builder = VGG16ModelBuilder()
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
