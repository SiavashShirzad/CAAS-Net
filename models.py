import tensorflow as tf
from tensorflow import keras


def conv_2d_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def transpose_skip_block(inputs, skip, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_2d_block(x, num_filters)
    return x


class DeepLabV3Builder(keras.Model):
    def __init__(self):
        super().__init__()

    def convolution_block(
            self,
            block_input,
            num_filters=256,
            kernel_size=3,
            dilation_rate=1,
            padding="same",
            use_bias=False,
    ):
        x = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
        )(block_input)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.ReLU()(x)

    def dilated_spatial_pyramid_pooling(self, dspp_input):
        dims = dspp_input.shape
        x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = self.convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = tf.keras.layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
        )(x)
        out_1 = self.convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
        x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = self.convolution_block(x, kernel_size=1)
        return output

    def deep_lab_v3_plus(self, image_size, num_classes):
        model_input = tf.keras.Input(shape=(image_size, image_size, 3))
        resnet = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
        x = resnet.get_layer("conv4_block6_2_relu").output
        x = self.dilated_spatial_pyramid_pooling(x)

        input_a = tf.keras.layers.UpSampling2D(
            size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
        input_b = resnet.get_layer("conv2_block3_2_relu").output
        input_b = self.convolution_block(input_b, num_filters=48, kernel_size=1)

        x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        x = self.convolution_block(x)
        x = self.convolution_block(x)
        x = tf.keras.layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
        model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
        return tf.keras.Model(inputs=model_input, outputs=model_output)

    def __call__(self, image_size, num_classes):
        return self.deep_lab_v3_plus(image_size=image_size, num_classes=num_classes)


class EfficientB0UnetBuilder(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        enb0 = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
        e1 = enb0.get_layer("input_1").output
        e2 = enb0.get_layer("block2a_expand_activation").output
        e3 = enb0.get_layer("block3a_expand_activation").output
        e4 = enb0.get_layer("block4a_expand_activation").output
        e5 = enb0.get_layer("block6a_expand_activation").output
        d1 = transpose_skip_block(e5, e4, 1024)
        d2 = transpose_skip_block(d1, e3, 512)
        d3 = transpose_skip_block(d2, e2, 256)
        d4 = transpose_skip_block(d3, e1, 128)
        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax")(d4)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientB0Unet")
        return model


class VGG16ModelBuilder(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

        e1 = vgg.get_layer("block1_conv2").output
        e2 = vgg.get_layer("block2_conv2").output
        e3 = vgg.get_layer("block3_conv3").output
        e4 = vgg.get_layer("block4_conv3").output
        e5 = vgg.get_layer("block5_conv3").output
        d1 = transpose_skip_block(e5, e4, 512)
        d2 = transpose_skip_block(d1, e3, 256)
        d3 = transpose_skip_block(d2, e2, 128)
        d4 = transpose_skip_block(d3, e1, 64)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax")(d4)
        model = tf.keras.models.Model(inputs, outputs, name="VGG16Unet")

        return model


class SimpleUnetBuilder(keras.Model):

    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))

        s1 = conv_2d_block(inputs, image_size / 8)
        e1 = tf.keras.layers.MaxPooling2D((2, 2))(s1)
        s2 = conv_2d_block(e1, image_size / 4)
        e2 = tf.keras.layers.MaxPooling2D((2, 2))(s2)
        s3 = conv_2d_block(e2, image_size / 2)
        e3 = tf.keras.layers.MaxPooling2D((2, 2))(s3)
        s4 = conv_2d_block(e3, image_size)
        e4 = tf.keras.layers.MaxPooling2D((2, 2))(s4)
        e5 = conv_2d_block(e4, 2 * image_size)

        d1 = transpose_skip_block(e5, s4, 512)
        d2 = transpose_skip_block(d1, s3, 256)
        d3 = transpose_skip_block(d2, s2, 128)
        d4 = transpose_skip_block(d3, s1, 64)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax")(d4)

        model = tf.keras.models.Model(inputs, outputs, name="SimpleUnet")
        return model


class ResNet50Builder(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        resnet50 = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

        e1 = resnet50.get_layer("input_1").output
        e2 = resnet50.get_layer("conv1_relu").output
        e3 = resnet50.get_layer("conv2_block3_out").output
        e4 = resnet50.get_layer("conv3_block4_out").output
        e5 = resnet50.get_layer("conv4_block6_out").output
        d1 = transpose_skip_block(e5, e4, 512)
        d2 = transpose_skip_block(d1, e3, 256)
        d3 = transpose_skip_block(d2, e2, 128)
        d4 = transpose_skip_block(d3, e1, 64)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax")(d4)
        model = tf.keras.models.Model(inputs, outputs, name="VGG16Unet")

        return model


class DenseNet121Unet(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        densenet121 = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)

        e1 = densenet121.get_layer("input_1").output
        e2 = densenet121.get_layer("conv1/relu").output
        e3 = densenet121.get_layer("pool2_relu").output
        e4 = densenet121.get_layer("pool3_relu").output
        e5 = densenet121.get_layer("pool4_relu").output
        d1 = transpose_skip_block(e5, e4, 512)
        d2 = transpose_skip_block(d1, e3, 256)
        d3 = transpose_skip_block(d2, e2, 128)
        d4 = transpose_skip_block(d3, e1, 64)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax")(d4)
        model = tf.keras.models.Model(inputs, outputs, name="DenseNet121")
        return model
