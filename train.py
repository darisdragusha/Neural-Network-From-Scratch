from data import train_labels, train_images,test_labels,test_images
from neuralNetwork import NeuralNetwork
import numpy as np

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

test_loss, test_accuracy = nn.evaluate(test_images, test_labels)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')