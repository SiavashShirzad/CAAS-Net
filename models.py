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


def transpose_skip_block_v2(inputs, skip, attention, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([x, skip, attention])
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
        model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation="softmax",
                                              name="multi")(x)
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

        d1 = transpose_skip_block(e5, e4, image_size*2)
        d2 = transpose_skip_block(d1, e3, image_size)
        d3 = transpose_skip_block(d2, e2, image_size//2)
        d4 = transpose_skip_block(d3, e1, image_size//4)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(d4)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientB0Unet")
        return model


class VGG16ModelBuilder(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

        e1 = base.get_layer("block1_conv2").output
        e2 = base.get_layer("block2_conv2").output
        e3 = base.get_layer("block3_conv3").output
        e4 = base.get_layer("block4_conv3").output
        e5 = base.get_layer("block5_conv3").output
        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size//2)
        d3 = transpose_skip_block(d2, e2, image_size//4)
        d4 = transpose_skip_block(d3, e1, image_size//8)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(d4)
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

        d1 = transpose_skip_block(e5, s4, image_size)
        d2 = transpose_skip_block(d1, s3, image_size//2)
        d3 = transpose_skip_block(d2, s2, image_size//4)
        d4 = transpose_skip_block(d3, s1, image_size//8)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(d4)

        model = tf.keras.models.Model(inputs, outputs, name="SimpleUnet")
        return model


class ResNet50Builder(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("conv1_relu").output
        e3 = base.get_layer("conv2_block3_out").output
        e4 = base.get_layer("conv3_block4_out").output
        e5 = base.get_layer("conv4_block6_out").output
        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size//2)
        d3 = transpose_skip_block(d2, e2, image_size//4)
        d4 = transpose_skip_block(d3, e1, image_size//8)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(d4)
        model = tf.keras.models.Model(inputs, outputs, name="VGG16Unet")

        return model


class DenseNet121Unet(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)

        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("conv1/relu").output
        e3 = base.get_layer("pool2_relu").output
        e4 = base.get_layer("pool3_relu").output
        e5 = base.get_layer("pool4_relu").output

        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size // 2)
        d3 = transpose_skip_block(d2, e2, image_size // 4)
        d4 = transpose_skip_block(d3, e1, image_size // 8)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(d4)
        model = tf.keras.models.Model(inputs, outputs, name="DenseNet121")
        return model


class ResNetRSUnet(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.resnet_rs.ResNetRS50(
            include_top=False,
            weights=None,
            input_tensor=inputs,
        )
        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("stem_1_stem_act_3").output
        e3 = base.get_layer("BlockGroup3__block_0__act_1").output
        e4 = base.get_layer("BlockGroup4__block_0__act_1").output
        e5 = base.get_layer("BlockGroup5__block_0__act_1").output

        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size // 2)
        d3 = transpose_skip_block(d2, e2, image_size // 4)
        d4 = transpose_skip_block(d3, e1, image_size // 8)

        outputs = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(d4)
        model = tf.keras.models.Model(inputs, outputs, name="ResnetRSUnet")
        return model


'''
UUNet are based on Unet architecture with backbone with an additional head. One of the decoding heads returns binary 
classification and the other head returns multi-class classification. Both heads' losses will affect the backbone's head 
in the backpropagation process.
'''


class DenseNet121UUnet(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)

        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("conv1/relu").output
        e3 = base.get_layer("pool2_relu").output
        e4 = base.get_layer("pool3_relu").output
        e5 = base.get_layer("pool4_relu").output

        # Multi class decoder of UU-Net
        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size // 2)
        d3 = transpose_skip_block(d2, e2, image_size // 4)
        d4 = transpose_skip_block(d3, e1, image_size // 8)

        # Binary class decoder of UU-Net
        bd1 = transpose_skip_block(e5, e4, image_size)
        bd2 = transpose_skip_block(bd1, e3, image_size // 2)
        bd3 = transpose_skip_block(bd2, e2, image_size // 4)
        bd4 = transpose_skip_block(bd3, e1, image_size // 8)

        # one head will predict the mask for all coronary arteries using sigmoid, and the other predicts classes
        output1 = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(
            d4)
        output2 = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="single")(
            bd4)
        model = tf.keras.models.Model(inputs, outputs=[output1, output2], name="DenseNet121")
        return model


class EfficientB0UUnetBuilder(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("block2a_expand_activation").output
        e3 = base.get_layer("block3a_expand_activation").output
        e4 = base.get_layer("block4a_expand_activation").output
        e5 = base.get_layer("block6a_expand_activation").output

        # Multi class decoder of UU-Net
        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size // 2)
        d3 = transpose_skip_block(d2, e2, image_size // 4)
        d4 = transpose_skip_block(d3, e1, image_size // 8)

        # Binary class decoder of UU-Net
        bd1 = transpose_skip_block(e5, e4, image_size)
        bd2 = transpose_skip_block(bd1, e3, image_size // 2)
        bd3 = transpose_skip_block(bd2, e2, image_size // 4)
        bd4 = transpose_skip_block(bd3, e1, image_size // 8)

        # one head will predict the mask for all coronary arteries using sigmoid, and the other predicts classes
        output1 = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(
            d4)
        output2 = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="single")(
            bd4)
        model = tf.keras.models.Model(inputs, outputs=[output1, output2], name="DenseNet121")
        return model


class ResNetRSUUnet(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.resnet_rs.ResNetRS50(
            include_top=False,
            weights=None,
            input_tensor=inputs,
        )
        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("stem_1_stem_act_3").output
        e3 = base.get_layer("BlockGroup3__block_0__act_1").output
        e4 = base.get_layer("BlockGroup4__block_0__act_1").output
        e5 = base.get_layer("BlockGroup5__block_0__act_1").output

        # Multi class decoder of UU-Net
        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size // 2)
        d3 = transpose_skip_block(d2, e2, image_size // 4)
        d4 = transpose_skip_block(d3, e1, image_size // 8)

        # Binary class decoder of UU-Net
        bd1 = transpose_skip_block(e5, e4, image_size)
        bd2 = transpose_skip_block(bd1, e3, image_size // 2)
        bd3 = transpose_skip_block(bd2, e2, image_size // 4)
        bd4 = transpose_skip_block(bd3, e1, image_size // 8)

        # one head will predict the mask for all coronary arteries using sigmoid, and the other predicts classes
        output1 = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(
            d4)
        output2 = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="single")(
            bd4)
        model = tf.keras.models.Model(inputs, outputs=[output1, output2], name="ResNetRSUUNet")
        return model


class ResNet50UUnetBuilder(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("conv1_relu").output
        e3 = base.get_layer("conv2_block3_out").output
        e4 = base.get_layer("conv3_block4_out").output
        e5 = base.get_layer("conv4_block6_out").output

        # Multi class decoder of UU-Net
        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size // 2)
        d3 = transpose_skip_block(d2, e2, image_size // 4)
        d4 = transpose_skip_block(d3, e1, image_size // 8)

        # Binary class decoder of UU-Net
        bd1 = transpose_skip_block(e5, e4, image_size)
        bd2 = transpose_skip_block(bd1, e3, image_size // 2)
        bd3 = transpose_skip_block(bd2, e2, image_size // 4)
        bd4 = transpose_skip_block(bd3, e1, image_size // 8)

        # one head will predict the mask for all coronary arteries using sigmoid, and the other predicts classes
        output1 = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(
            d4)
        output2 = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="single")(
            bd4)
        model = tf.keras.models.Model(inputs, outputs=[output1, output2], name="ResNetUUnet")
        return model


'''
another implementation of U-Net called W-Net. 
We used it to first detect all coronary arteries and then anatomically classify them.
'''


class SimpleWnetBuilder(keras.Model):

    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))

        # first part of encoder-decoder to segment all coronary areteries
        s1 = conv_2d_block(inputs, image_size / 8)
        e1 = tf.keras.layers.MaxPooling2D((2, 2))(s1)
        s2 = conv_2d_block(e1, image_size / 4)
        e2 = tf.keras.layers.MaxPooling2D((2, 2))(s2)
        s3 = conv_2d_block(e2, image_size / 2)
        e3 = tf.keras.layers.MaxPooling2D((2, 2))(s3)
        s4 = conv_2d_block(e3, image_size)
        e4 = tf.keras.layers.MaxPooling2D((2, 2))(s4)
        e5 = conv_2d_block(e4, 2 * image_size)

        d1 = transpose_skip_block(e5, s4, image_size)
        d2 = transpose_skip_block(d1, s3, image_size//2)
        d3 = transpose_skip_block(d2, s2, image_size//4)
        d4 = transpose_skip_block(d3, s1, image_size//8)

        output1 = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="single")(d4)

        # Second part of the model to anatomically classify first part's output
        s2_1 = conv_2d_block(output1, image_size / 8)
        e2_1 = tf.keras.layers.MaxPooling2D((2, 2))(s2_1)
        s2_2 = conv_2d_block(e2_1, image_size / 4)
        e2_2 = tf.keras.layers.MaxPooling2D((2, 2))(s2_2)
        s2_3 = conv_2d_block(e2_2, image_size / 2)
        e2_3 = tf.keras.layers.MaxPooling2D((2, 2))(s2_3)
        s2_4 = conv_2d_block(e2_3, image_size)
        e2_4 = tf.keras.layers.MaxPooling2D((2, 2))(s2_4)
        e2_5 = conv_2d_block(e2_4, 2 * image_size)

        d2_1 = transpose_skip_block(e2_5, s2_4, image_size)
        d2_2 = transpose_skip_block(d2_1, s2_3, image_size//2)
        d2_3 = transpose_skip_block(d2_2, s2_2, image_size//4)
        d2_4 = transpose_skip_block(d2_3, s2_1, image_size//8)

        output2 = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multiple")(d2_4)

        model = tf.keras.models.Model(inputs, outputs=[output1, output2], name="SimpleUnet")
        return model


class AttentionEfficientWNet(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=inputs,
        )
        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("block2a_expand_activation").output
        e3 = base.get_layer("block3a_expand_activation").output
        e4 = base.get_layer("block4a_expand_activation").output
        e5 = base.get_layer("block6a_expand_activation").output

        # Binary class decoder of TridentNet
        bd1 = transpose_skip_block(e5, e4, image_size)
        bd2 = transpose_skip_block(bd1, e3, image_size // 2)
        bd3 = transpose_skip_block(bd2, e2, image_size // 4)
        bd4 = transpose_skip_block(bd3, e1, image_size // 8)

        # Multi class decoder of TridentNet
        d1 = transpose_skip_block_v2(e5, e4, bd1, image_size)
        d2 = transpose_skip_block_v2(d1, e3, bd2, image_size // 2)
        d3 = transpose_skip_block_v2(d2, e2, bd3, image_size // 4)
        d4 = transpose_skip_block_v2(d3, e1, bd4, image_size // 8)

        # one head will predict the mask for all coronary arteries using sigmoid, and the other predicts classes
        output1 = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(
            d4)
        output2 = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="single")(
            bd4)
        model = tf.keras.models.Model(inputs, outputs=[output1, output2], name="AttentionEfficientWNet")
        return model


# Multi-task learning
# A new model named trident-Net

class ResNetRSTridentNet(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.resnet_rs.ResNetRS50(
            include_top=False,
            weights=None,
            input_tensor=inputs,
        )
        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("stem_1_stem_act_3").output
        e3 = base.get_layer("BlockGroup3__block_0__act_1").output
        e4 = base.get_layer("BlockGroup4__block_0__act_1").output
        e5 = base.get_layer("BlockGroup5__block_0__act_1").output

        # Multi class decoder of TridentNet
        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size // 2)
        d3 = transpose_skip_block(d2, e2, image_size // 4)
        d4 = transpose_skip_block(d3, e1, image_size // 8)

        # Binary class decoder of TridentNet
        bd1 = transpose_skip_block(e5, e4, image_size)
        bd2 = transpose_skip_block(bd1, e3, image_size // 2)
        bd3 = transpose_skip_block(bd2, e2, image_size // 4)
        bd4 = transpose_skip_block(bd3, e1, image_size // 8)

        # Classifier head
        X = base.get_layer("BlockGroup5__block_2__output_act").output
        X = keras.layers.MaxPooling2D()(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Dense(32, activation="relu")(X)
        X = keras.layers.Dense(16, activation="relu")(X)

        # one head will predict the mask for all coronary arteries using sigmoid, and the other predicts classes
        output1 = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(
            d4)
        output2 = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="single")(
            bd4)
        output3 = tf.keras.layers.Dense(7, activation="softmax", name="classifier")(
            X)
        model = tf.keras.models.Model(inputs, outputs=[output1, output2, output3], name="ResNetRSTridentNet")
        return model


class EfficientTridentNet(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=inputs,
        )
        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("block2a_expand_activation").output
        e3 = base.get_layer("block3a_expand_activation").output
        e4 = base.get_layer("block4a_expand_activation").output
        e5 = base.get_layer("block6a_expand_activation").output

        # Multi class decoder of TridentNet
        d1 = transpose_skip_block(e5, e4, image_size)
        d2 = transpose_skip_block(d1, e3, image_size // 2)
        d3 = transpose_skip_block(d2, e2, image_size // 4)
        d4 = transpose_skip_block(d3, e1, image_size // 8)

        # Binary class decoder of TridentNet
        bd1 = transpose_skip_block(e5, e4, image_size)
        bd2 = transpose_skip_block(bd1, e3, image_size // 2)
        bd3 = transpose_skip_block(bd2, e2, image_size // 4)
        bd4 = transpose_skip_block(bd3, e1, image_size // 8)

        # Classifier head
        X = base.get_layer("top_activation").output
        X = keras.layers.MaxPooling2D()(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Dense(32, activation="relu")(X)
        X = keras.layers.Dense(16, activation="relu")(X)

        # one head will predict the mask for all coronary arteries using sigmoid, and the other predicts classes
        output1 = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(
            d4)
        output2 = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="single")(
            bd4)
        output3 = tf.keras.layers.Dense(7, activation="softmax", name="classifier")(
            X)
        model = tf.keras.models.Model(inputs, outputs=[output1, output2, output3], name="EfficientTridentNet")
        return model


class AttentionEfficientTridentNet(keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, image_size, number_classes):
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=inputs,
        )
        e1 = base.get_layer("input_1").output
        e2 = base.get_layer("block2a_expand_activation").output
        e3 = base.get_layer("block3a_expand_activation").output
        e4 = base.get_layer("block4a_expand_activation").output
        e5 = base.get_layer("block6a_expand_activation").output

        # Binary class decoder of TridentNet
        bd1 = transpose_skip_block(e5, e4, image_size)
        bd2 = transpose_skip_block(bd1, e3, image_size // 2)
        bd3 = transpose_skip_block(bd2, e2, image_size // 4)
        bd4 = transpose_skip_block(bd3, e1, image_size // 8)

        # Multi class decoder of TridentNet
        d1 = transpose_skip_block_v2(e5, e4, bd1, image_size)
        d2 = transpose_skip_block_v2(d1, e3, bd2, image_size // 2)
        d3 = transpose_skip_block_v2(d2, e2, bd3, image_size // 4)
        d4 = transpose_skip_block_v2(d3, e1, bd4, image_size // 8)

        # Classifier head
        X = base.get_layer("top_activation").output
        X = keras.layers.MaxPooling2D()(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Dense(32, activation="relu")(X)
        X = keras.layers.Dense(16, activation="relu")(X)

        # one head will predict the mask for all coronary arteries using sigmoid, and the other predicts classes
        output1 = tf.keras.layers.Conv2D(number_classes, 1, padding="same", activation="softmax", name="multi")(
            d4)
        output2 = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="single")(
            bd4)
        output3 = tf.keras.layers.Dense(7, activation="softmax", name="classifier")(
            X)
        model = tf.keras.models.Model(inputs, outputs=[output1, output2, output3], name="AttentionEfficientTridentNet")
        return model
