import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Randomly initialize weights and biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))
        
    def forward(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output
    
    def backward(self, X, y, learning_rate=0.1):
        # Backpropagation
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases using gradient descent
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# Create a simple function to train the neural network
def train_nn(model, X_train, y_train, epochs=10000, learning_rate=0.1):
    for epoch in range(epochs):
        model.forward(X_train)
        model.backward(X_train, y_train, learning_rate)

        if epoch % 1000 == 0:
            loss = np.mean(np.square(y_train - model.final_output))
            print(f"Epoch {epoch}, Loss: {loss}")

# Prepare training data (X: Board states, y: Outputs for winning moves)
# Each state of the board will be represented as a vector of 9 elements (X, O, Empty)
# We will need to construct a dataset of board states and the corresponding output (win or not).

# Example Tic-Tac-Toe dataset:
# Board (3x3) -> Flattened to 9 values, where X = 1, O = -1, Empty = 0
# Output: 1 if it's a winning move, 0 if it's not.

X_train = np.array([
    [ 1,  0, -1,  0,  1, -1,  0,  0,  0],  # Example board state
    [ 1,  1,  1,  0, -1, -1,  0,  0,  0],  # Another example board state
    # Add more board states as required...
])

y_train = np.array([
    [1],  # Winning move
    [0],  # Not a winning move
    # Corresponding outputs...
])

# Initialize the neural network
input_size = 9  # Flattened 3x3 Tic-Tac-Toe board
hidden_size = 18  # Arbitrary size for hidden layer
output_size = 1  # Binary output (win or not)

model = NeuralNetwork(input_size, hidden_size, output_size)

# Train the network
train_nn(model, X_train, y_train, epochs=10000, learning_rate=0.1)

# Testing the trained model
test_input = np.array([[1, 0, -1, 0, 0, -1, 1, 0, 0]])  # Example test board state
prediction = model.forward(test_input)
print(f"Prediction for test input: {prediction}")
