import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        # weight initialization using Xavier Initialization
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        
        # bias initialization  
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1,output_size))

    def sigmoid_activation_function(self, x):
        return 1/(1+np.exp(-x))
    
    def relu_activation_function(self, x):
        return max(0,x)
    
    def leaky_relu_activation_function(self, x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def softmax_activation_function(self, x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum(axis=1, keepdims=True)  # Normalize to get probabilities
    
    def feedforward(self, input_data):
        # hidden layer 
        self.hidden_layer = np.dot(input_data , self.weights_input_hidden) + self.bias_hidden
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
    
    def backpass(self, input_data, y_true):
        # because i am using softmax AF and cross-entropy loss:
        output_error = self.output_layer_activation - y_true

        # calculates the gradient of each weight between the hidden and output layer
        gradient_weights_hidden_output = np.dot(self.hidden_layer_activation.T,output_error)

        # output neuron bias gradient
        gradient_bias_output = np.sum(self.bias_output,axis=0, keepdims=True)
        # propagate the error to the hidden layer
        hidden_layer_error = np.dot(output_error, self.weights_hidden_output.T) * self.leaky_relu_derivative(self.hidden_layer)

        # hidden neuron bias gradient
        gradient_bias_hidden = np.sum(hidden_layer_error, axis=0, keepdims=True)
        # calculate the gradient of each weight between the input and hidden layer
        gradient_weights_input_hidden = np.dot(input_data.T, hidden_layer_error)

        # store the weights and biasis
        self.gradient_weights_hidden_output = gradient_weights_hidden_output
        self.gradient_weights_input_hidden = gradient_weights_input_hidden
        self.gradient_bias_output = gradient_bias_output
        self.gradient_bias_hidden = gradient_bias_hidden

    def adaptive_learning_rate(self, initial_learning_rate, gradient, epsilon=1e-8):
        # adaptive learning rate based on the magnitude of the gradient(slope) - CHATGPT
        return initial_learning_rate / (np.sqrt(np.sum(np.square(gradient))) + epsilon)
    
    def gradient_descent(self, learning_rate):
        # store adaptive learning rates
        adaptive_lr_input_hidden = self.adaptive_learning_rate(learning_rate, self.gradient_weights_input_hidden)
        adaptive_lr_hidden_output = self.adaptive_learning_rate(learning_rate, self.gradient_weights_hidden_output)
    
        # adjust weights and biases based on learning rate and gradients
        self.weights_hidden_output -= adaptive_lr_hidden_output * self.gradient_weights_hidden_output
        self.weights_input_hidden -= adaptive_lr_input_hidden *self.gradient_weights_input_hidden
        self.bias_hidden -= adaptive_lr_input_hidden * self.gradient_bias_hidden
        self.bias_output -= adaptive_lr_hidden_output * self.gradient_bias_output

    def train(self, input_data, epochs,y_true, batch_size, learning_rate):
        
        num_samples = input_data.shape[0]

        for epoch in range(epochs):
            # shuffle the data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            input_data = input_data[indices]
            y_true = y_true[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = input_data[i:i + batch_size]
                y_batch = y_true[i:i + batch_size]

                # forward pass
                self.feedforward(input_data=X_batch)

                #calculate loss
                loss = self.cross_entropy_loss(y_true=y_batch,y_pred=self.output_layer_activation)

                # back pass(calculate gradient)
                self.backpass(input_data=X_batch,y_true=y_batch)

                # update weights and biases 
                self.gradient_descent(learning_rate)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    def save_model(self, filename):
        """Save the weights and biases to a file."""
        np.savez(filename,
                 weights_input_hidden=self.weights_input_hidden,
                 weights_hidden_output=self.weights_hidden_output,
                 bias_hidden=self.bias_hidden,
                 bias_output=self.bias_output)

    def load_model(self, filename):
        """Load the weights and biases from a file."""
        data = np.load(filename)
        self.weights_input_hidden = data['weights_input_hidden']
        self.weights_hidden_output = data['weights_hidden_output']
        self.bias_hidden = data['bias_hidden']
        self.bias_output = data['bias_output']

    def evaluate(self, test_images, test_labels):

        predictions = self.predict(test_images)

        # Calculate accuracy
        accuracy = np.mean(predictions == test_labels)
        
        test_labels_one_hot = self.one_hot_encode(test_labels, num_classes=10)
        loss = self.cross_entropy_loss(test_labels_one_hot, self.predict_proba(test_images))

        return loss, accuracy

    def one_hot_encode(self, y, num_classes):
        return np.eye(num_classes)[y]
    def predict(self, X):

        output = self.feedforward(X)
        # Get the predicted class by finding the index of the maximum value in each output
        predicted_classes = np.argmax(output, axis=1)

        return predicted_classes

    def predict_proba(self, X):
        """
        Get the predicted probabilities for each class.

        Parameters:
        - X: Input data (shape: num_samples x input_size)

        Returns:
        - output: Probabilities for each class (shape: num_samples x num_classes)
        """
        # Forward pass to get the output probabilities
        output = self.feedforward(X)  # This should give the output of the last layer (before argmax)
        return output