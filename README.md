# simple-auto-encoder
A simple convolutional auto-encoder implemented using Keras and Tensorflow

Usage: "python main.py <input_dir> [option 1] [option 2]..."

Run "python main.py --help" for a detailed help message:

usage: main.py [-h] [--compression_level COMPRESSION_LEVEL]
               [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE]
               [--train_ratio TRAIN_RATIO]
               [--mode {visualize,train,encode,decode}]
               input_folder

Input, network configuration and other settings of the auto-encoder

positional arguments:
  input_folder          Input folder for the network, should contain color
                        images of size 64x64

optional arguments:
  -h, --help            show this help message and exit
  --compression_level COMPRESSION_LEVEL
                        How much to compress the image (a fraction of the
                        original size, eg. 0.05)
  --n_epochs N_EPOCHS   The number of epochs in the training stage of the
                        network
  --batch_size BATCH_SIZE
                        The batch size in the training stage of the network
  --train_ratio TRAIN_RATIO
                        The train ratio out of the entire data set
  --mode {visualize,train,encode,decode}
                        Mode. Can be 'visualize', 'train', 'encode' or
                        'decode'. 'visualize' is used to assess the
                        performance of the network, 'train' to train the
                        network, and 'encode' and 'decode' are used to encode
                        the images in the input folder and decode them,
                        respectively
