############ imports ###############
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from auto_encoder import ConvolutionalAutoEncoder

############ constants ###############

VIS_MODE = "visualize"
TRAIN_MODE = "train"
ENCODE_MODE = "encode"
DECODE_MODE = "decode"

DEFAULT_COMPRESSION_LEVEL = 0.01
DEFAULT_EPOCH_NUMBER = 200
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODE = VIS_MODE
IMAGE_SIZE = 64
NUMBER_OF_CHANNELS = 3

ENCODED_FOLDER = "encoded/"
DECODED_FOLDER = "decoded/"
MODEL_FOLDER = "model/"

############ the main module ###############

def parse_args():
    """
    Parses and validates the command line arguments
    :return: ArgumentParser object with the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Input, network configuration and other settings of the auto-encoder')
    parser.add_argument('input_folder', type=str,
                        help="Input folder for the network, should contain color images of size 64x64")
    parser.add_argument('--compression_level', type=float, default=DEFAULT_COMPRESSION_LEVEL,
                        help="How much to compress the image (a fraction of the original size, eg. 0.05)")
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_EPOCH_NUMBER,
                        help="The number of epochs in the training stage of the network")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help="The batch size in the training stage of the network")
    parser.add_argument('--train_ratio', type=float, default=DEFAULT_TRAIN_RATIO,
                        help="The train ratio out of the entire data set")
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE,
                        choices=['visualize', 'train', 'encode', 'decode'],
                        help="""Mode. Can be 'visualize', 'train', 'encode' or 'decode'. 'visualize' is used to assess
                             the performance of the network, 'train' to train the network, and 'encode' and 'decode'
                             are used to encode the images in the input folder and decode them, respectively""")

    args = parser.parse_args()
    print(args)

    # validates the arguments
    if not os.path.exists(args.input_folder):
        raise argparse.ArgumentError("Input folder doesn't exist")

    if args.compression_level < 0.0 or args.compression_level > 1.0:
        raise argparse.ArgumentError("Compression level must be between 0.0 and 1.0")

    # creates needed folders, if they don't exist
    if not os.path.exists(os.path.join(MODEL_FOLDER)):
        os.mkdir(os.path.join(MODEL_FOLDER))
        os.mkdir(os.path.join(DECODED_FOLDER))
        os.mkdir(os.path.join(ENCODED_FOLDER))

    return args


def load_image_data(args, for_train=True):
    """
    Prepossesses and prepares the input data, shuffle it and split it to train and test sets
    :param args: the parsed command line arguments
    :param for_train: if True, the loaded data will be shuffled and split into train and test sets
    :return: if for_train is True, Two numpy arrays of shape
    [number_of_images, IMAGE_SIZE, IMAGE_SIZE, NUMBER_OF_CHANNELS]. Else - One numpy array of the same shape
    """
    number_of_images = len(os.listdir(args.input_folder))
    data_set = np.empty((number_of_images, IMAGE_SIZE, IMAGE_SIZE, NUMBER_OF_CHANNELS))

    # loads the data set
    for i, filename in enumerate(os.listdir(args.input_folder)):
        image = plt.imread(os.path.join(args.input_folder, filename))
        image = image[:,:,:3]  # I used RGBA images for the training, so I use this line to remove the 4th channel
        data_set[i] = image

    # shuffles it and splits it to train and test sets, if needed
    if for_train:
        np.random.shuffle(data_set)
        train_set = data_set[:ceil(number_of_images * args.train_ratio)]
        test_set = data_set[ceil(number_of_images * args.train_ratio):]
        return train_set, test_set
    else:
        return data_set


def save_encoded_images(args, encoded_images):
    """
    Saves the encoded images into the folder ENCODED_FOLDER
    :param args: the parsed command line arguments
    :param encoded_images: a numpy array of shape [num_of_images, latent_space_size]
    """
    for i, filename in enumerate(os.listdir(args.input_folder)):  # goes over the the input dit to get the images' names
        new_filename = filename[:filename.find(".png")]  # removes the extension
        np.save(ENCODED_FOLDER + new_filename, encoded_images[i])


def load_encoded_images(code_size):
    """
    Loads the encoded images from the folder ENCODED_FOLDER.
    :param latent_space_size: the size of the latent space
    :return: Numpy array of shape [num_of_images, latent_space_size]
    """
    number_of_images = len(os.listdir(ENCODED_FOLDER))
    data_set = np.empty((number_of_images, code_size))
    for i, filename in enumerate(os.listdir(ENCODED_FOLDER)):
        encoded_image = np.load(os.path.join(ENCODED_FOLDER, filename))
        data_set[i] = encoded_image

    return data_set


def save_decoded_images(images):
    """
    Saves the decoded images into the DECODED_FOLDER folder.
    :param images: numpy array of shape [number_of_images, IMAGE_SIZE, IMAGE_SIZE, NUMBER_OF_CHANNELS] representing the
    images.
    """
    for i, filename in enumerate(os.listdir(ENCODED_FOLDER)):
        new_filename = filename[:filename.find(".npy")]  # removes the ".npy" extension
        plt.imsave(DECODED_FOLDER + new_filename + ".png", images[i], format="png")


def visualize_compression(auto_encoder, images):
    """
    Visualizes the compression and extraction abilities of the network
    :param auto_encoder: the auto encoder model
    :param images: numpy array of shape [number_of_images, IMAGE_SIZE, IMAGE_SIZE, NUMBER_OF_CHANNELS] representing the
    images.
    """
    encoded = auto_encoder.encode_images(images)
    decoded = auto_encoder.decode_images(encoded)

    l_size = auto_encoder.latent_space_size
    root_l_size = int(l_size ** 0.5)

    # visualizing the results
    num_of_samples_to_display = min(10, images.shape[0])
    for i in range(num_of_samples_to_display):
        # original image
        plt.subplot(3, num_of_samples_to_display, i + 1)
        plt.imshow(images[i])

        # the encoding
        plt.subplot(3, num_of_samples_to_display, i + num_of_samples_to_display + 1)
        plt.imshow(encoded[i][:root_l_size ** 2].reshape((root_l_size, root_l_size)))  # just an ugly way to visualize
                                                                                       # the encoding
        plt.gray()

        # the reconstruction
        plt.subplot(3, num_of_samples_to_display, i + 2 * num_of_samples_to_display + 1)
        plt.imshow(decoded[i])

    plt.show()


def visualize(args):
    """
    Visualizes the compression and extraction abilities of the network
    :param args: the parsed command line arguments
    """
    encoder = ConvolutionalAutoEncoder(args.compression_level)
    train_set, test_set = load_image_data(args)
    encoder.train(train_set, test_set, args.n_epochs, args.batch_size)
    visualize_compression(encoder, train_set)


def train_network(args):
    """
    Trains the network and saves the model to MODEL_FOLDER folder
    :param args: the parsed command line arguments
    """
    encoder = ConvolutionalAutoEncoder(args.compression_level)
    train_set, test_set = load_image_data(args)
    encoder.train(train_set, test_set, args.n_epochs, args.batch_size)
    encoder.save_model(MODEL_FOLDER)


def encode_images(args):
    """
    Encodes the images in the input folder using an already trained model, and saves the encoded
     images in ENCODED_FOLDER
    :param args:  the parsed command line arguments
    """
    images = load_image_data(args, for_train=False)
    encoder = ConvolutionalAutoEncoder(args.compression_level, define_structure=False)
    encoder.restore_model(MODEL_FOLDER)
    encoded_images = encoder.encode_images(images)
    save_encoded_images(args, encoded_images)


def decode_images(args):
    """
    Decode the images in ENCODED_FOLDER using an already trained model, and saves the decoded
     images in DECODED_FOLDER
    :param args: the parsed command line arguments
    """
    encoder = ConvolutionalAutoEncoder(args.compression_level, define_structure=False)
    encoder.restore_model(MODEL_FOLDER)
    encoded_images = load_encoded_images(encoder.latent_space_size)
    decoded_images = encoder.decode_images(encoded_images)
    save_decoded_images(decoded_images)


def main():
    """
    The main module
    """
    args = parse_args()
    if args.mode == VIS_MODE:
        visualize(args)
    elif args.mode == TRAIN_MODE:
        train_network(args)
    elif args.mode == ENCODE_MODE:
        encode_images(args)
    else:
        decode_images(args)

if __name__ == "__main__":
    main()