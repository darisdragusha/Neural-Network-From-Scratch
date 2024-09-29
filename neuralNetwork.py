import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        # weight initialization using Xavier Initialization
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        
        # bias initialization using 
        self.bias_hidden = np.zeros(1, hidden_size)
        self.bias_output = np.zeros(1,output_size)

    def sigmoid_activation_function(self, x):
        return 1/(1+np.exp(-x))
    
    def relu_activation_function(self, x):
        return max(0,x)
    
    def leaky_relu_activation_function(self, x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)
    
    def softmax_activation_function(x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum(axis=1, keepdims=True)  # Normalize to get probabilities
    
    def feedforward(self, input_data):
        # hidden layer 
        self.hidden_layer = np.dot(input_data + self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_activation = self.leaky_relu_activation_function(self.hidden_layer)

        #output layer
        self.output_layer = np.dot(self.hidden_layer_activation,self.weights_hidden_output)+self.bias_output
        self.output_layer_activation = self.softmax_activation_function(self.output_layer)

        return self.output_layer_activation
    

    def cross_entropy_loss(self, y_true, y_pred):
        # Clip predictions to prevent log(0) and ensure numerical stability
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        
        # Calculate cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    