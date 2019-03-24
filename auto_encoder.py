############ imports ###############
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.python.keras.models import Model, load_model

############ constants ###############

IMAGE_SIZE = 64
NUMBER_OF_CHANNELS = 3
CONVOLUTION_KERNEL_SIZE = (3, 3)
POOLING_SIZE = (2, 2)
AUTO_ENCODER_MODEL_NAME = "auto_encoder"
ENCODER_MODEL_NAME = "encoder"
DECODER_MODEL_NAME = "decoder"

############ class definition ###############

class ConvolutionalAutoEncoder(object):
    """
    Implementation of a convolutional auto-encoder using Tensorflow and Keras.
    """

    def __init__(self, compression_level, define_structure=True):
        """
        Initializes the auto-encoder.
        :param compression_level: How much to compress the image (a fraction of the original size, eg. 0.25).
        """
        self.latent_space_size = int((IMAGE_SIZE * IMAGE_SIZE) * compression_level)

        if define_structure:
            # defining the network structure
            input_layer = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUMBER_OF_CHANNELS))
            code_layer = self._build_encoder(input_layer, self.latent_space_size)
            output_layer = self._build_decoder(code_layer)

            # defining the models - full model and the encoder
            self._auto_encoder_model = Model(input_layer, output_layer)
            self._encoder_model = Model(input_layer, code_layer)

            # defining the decoder model (in the best way I could find)
            encoded_input = Input(shape=(self.latent_space_size,))
            layer = self._auto_encoder_model.layers[-10](encoded_input)
            for l in range(9, 0, -1):
                layer = self._auto_encoder_model.layers[-l](layer)
            self._decoder_model = Model(encoded_input, layer)

            # compiling the model
            self._auto_encoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    def _build_encoder(self, input_layer, code_size):
        """
        Builds a 4-layer-convolutional encoder.
        """
        conv_layer_1 = Conv2D(16, CONVOLUTION_KERNEL_SIZE, activation='relu', padding='same')(input_layer)
        pooling_layer_1 = MaxPooling2D(POOLING_SIZE, padding='same')(conv_layer_1)
        conv_layer_2 = Conv2D(32, CONVOLUTION_KERNEL_SIZE, activation='relu', padding='same')(pooling_layer_1)
        pooling_layer_2 = MaxPooling2D(POOLING_SIZE, padding='same')(conv_layer_2)
        conv_layer_3 = Conv2D(64, CONVOLUTION_KERNEL_SIZE, activation='relu', padding='same')(pooling_layer_2)
        pooling_layer_3 = MaxPooling2D(POOLING_SIZE, padding='same')(conv_layer_3)
        conv_layer_4 = Conv2D(128, CONVOLUTION_KERNEL_SIZE, activation='relu', padding='same')(pooling_layer_3)
        pooling_layer_4 = MaxPooling2D(POOLING_SIZE, padding='same')(conv_layer_4)
        flatted_layer = Flatten()(pooling_layer_4)
        code_layer = Dense(code_size, activation='relu')(flatted_layer)
        return code_layer

    def _build_decoder(self, code_layer):
        """
        Builds a 4-layer-convolutional decoder.
        """
        dense_layer = Dense(2048, activation='relu')(code_layer)
        reshape_layer = Reshape((4, 4, 128))(dense_layer)
        upsampling_layer_1 = UpSampling2D(POOLING_SIZE)(reshape_layer)
        conv_layer_1 = Conv2D(64, CONVOLUTION_KERNEL_SIZE, activation='relu', padding='same')(upsampling_layer_1)
        upsampling_layer_2 = UpSampling2D(POOLING_SIZE)(conv_layer_1)
        conv_layer_2 = Conv2D(32, CONVOLUTION_KERNEL_SIZE, activation='relu', padding='same')(upsampling_layer_2)
        upsampling_layer_3 = UpSampling2D(POOLING_SIZE)(conv_layer_2)
        conv_layer_3 = Conv2D(16, CONVOLUTION_KERNEL_SIZE, activation='relu', padding='same')(upsampling_layer_3)
        upsampling_layer_4 = UpSampling2D(POOLING_SIZE)(conv_layer_3)
        output_layer = Conv2D(3, CONVOLUTION_KERNEL_SIZE, activation='sigmoid', padding='same')(upsampling_layer_4)
        return output_layer

    def train(self, train_data, test_data, n_epochs, batch_size):
        """
        Trains the auto-encoder model
        :param train_data: The training set, of shape [set_size, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE]
        :param test_data: The testing set, of shape [set_size, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE]
        :param n_epochs: number of epochs to train
        :param batch_size: number of samples in each training batch
        """
        self._auto_encoder_model.fit(train_data, train_data,
                                     epochs=n_epochs,
                                     batch_size=batch_size,
                                     validation_data=(test_data, test_data))

    def encode_images(self, images):
        """
        Encodes a set of images to a compact form
        :param images: a sequence of images with shape [num_of_images, IMAGE_SIZE, IMAGE_SIZE, NUMBER_OF_CHANNELS]
        :return: a sequence of encoded images of shape [num_of_images, self.latent_space_size]
        """
        return self._encoder_model.predict(images)

    def decode_images(self, encoded_images):
        """
        Decodes a set of compressed images
        :param encoded_images: a sequence of encoded images with shape [num_of_images, self.latent_space_size]
        :return: a sequence of decoded images with shape [num_of_images, IMAGE_SIZE, IMAGE_SIZE, NUMBER_OF_CHANNELS]
        """
        return self._decoder_model.predict(encoded_images)

    def encode_and_decode_images(self, image):
        """
        Compresses and then extracts a set of images
        :param image: a sequence of images with shape [num_of_images, IMAGE_SIZE, IMAGE_SIZE, NUMBER_OF_CHANNELS]
        :return: a sequence of images with shape [num_of_images, IMAGE_SIZE, IMAGE_SIZE, NUMBER_OF_CHANNELS]
        """
        return self._auto_encoder_model.predict(image)

    def save_model(self, path):
        """
        Saves the model to a file.
        :param path: the path to the folder in which the model will be saved.
        """
        self._auto_encoder_model.save(path + AUTO_ENCODER_MODEL_NAME + ".h5")
        self._encoder_model.save(path + ENCODER_MODEL_NAME + ".h5")
        self._decoder_model.save(path + DECODER_MODEL_NAME + ".h5")

    def restore_model(self, path):
        """
        Restores the model from a folder
        :param path: the path to the folder in which the model is saved.
        """
        self._auto_encoder_model = load_model(path + AUTO_ENCODER_MODEL_NAME + ".h5")
        self._encoder_model = load_model(path + ENCODER_MODEL_NAME + ".h5")
        self._decoder_model = load_model(path + DECODER_MODEL_NAME + ".h5")