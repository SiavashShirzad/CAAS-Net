from models import DeepLabV3Builder, EfficientB0UnetBuilder, VGG16ModelBuilder, SimpleUnetBuilder, ResNet50Builder, \
    DenseNet121Unet, DenseNet121UUnet, SimpleWnetBuilder, AttentionEfficientWNet, EfficientB0UUnetBuilder, \
    ResNet50UUnetBuilder, ResNetRSUnet, ResNetRSUUnet, ResNetRSTridentNet, EfficientTridentNet,\
    AttentionEfficientTridentNet, AttentionDenseWNet
import tensorflow as tf


class ModelBuilder:

    def __init__(self, image_size, number_classes, model_name):
        self.image_size = image_size
        self.number_classes = number_classes
        self.model_name = model_name

        if self.model_name == "deeplab":
            model_builder = DeepLabV3Builder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "AttentionEfficientTridentNet":
            model_builder = AttentionEfficientTridentNet()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "EfficientB0Unet":
            model_builder = EfficientB0UnetBuilder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "vgg16":
            model_builder = VGG16ModelBuilder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "SimpleUnet":
            model_builder = SimpleUnetBuilder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "ResnetUnet":
            model_builder = ResNet50Builder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "DenseNet121":
            model_builder = DenseNet121Unet()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "DenseNetUUnet":
            model_builder = DenseNet121UUnet()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "SimpleWnet":
            model_builder = SimpleWnetBuilder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "AttentionEfficientWNet":
            model_builder = AttentionEfficientWNet()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "AttentionDenseWNet":
            model_builder = AttentionDenseWNet()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "EfficientUUnet":
            model_builder = EfficientB0UUnetBuilder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "ResNetUUnet":
            model_builder = ResNet50UUnetBuilder()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "ResNetRSUnet":
            model_builder = ResNetRSUnet()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "ResNetRSUUnet":
            model_builder = ResNetRSUUnet()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "ResNetRSTridentNet":
            model_builder = ResNetRSTridentNet()
            self.model = model_builder(self.image_size, self.number_classes)

        if self.model_name == "EfficientTridentNet":
            model_builder = EfficientTridentNet()
            self.model = model_builder(self.image_size, self.number_classes)

    def summary(self):
        return self.model.summary()

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, dataset, validation_data=None, epochs=100, callbacks=None):
        return self.model.fit(dataset, validation_data=validation_data, epochs=epochs, callbacks=callbacks)

    def evaluate(self, data):
        return self.model.evaluate(data)

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, data):
        return self.model.evaluate(data)

    def load_best(self, weight_path):
        self.model.load_weights(weight_path).expect_partial()
        print("Model is loaded.")

    def save(self, path):
        tf.keras.models.save_model(self.model, path, include_optimizer=False)
        print("model is saved")

    def visualize(self, save_path: str):
        tf.keras.utils.plot_model(
            self.model,
            to_file=save_path + '.png',
            dpi=600,
        )
