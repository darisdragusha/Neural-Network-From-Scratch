import tensorflow as tf
from neuralNetwork import NeuralNetwork
import numpy as np

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images (28x28 becomes 784)
train_images = train_images.reshape(-1, 28 * 28)
test_images = test_images.reshape(-1, 28 * 28)

input_size = train_images.shape[1]
hidden_size = 64
output_size = 10

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

train_labels_one_hot = one_hot_encode(train_labels, num_classes=10)

nn = NeuralNetwork(input_size,hidden_size,output_size)

learning_rate = 0.01
epochs = 75
batch_size = 16

nn.train(train_images,epochs,train_labels_one_hot, batch_size,learning_rate)

nn.save_model("trained_model.npz")
