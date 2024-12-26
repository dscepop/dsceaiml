import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input and expected output for AND gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [0], [0], [1]])

# Hyperparameters
learning_rate = 0.1
epochs = 10000
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Initialize weights and biases
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_input)

    # Backpropagation
    error = labels - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    d_hidden_layer = d_predicted_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Final predicted output
print("\nFinal predicted output:")
print(predicted_output)

OUTPUT:
Epoch 0, Error: 0.6018209938312007
Epoch 1000, Error: 0.3604007367473875
Epoch 2000, Error: 0.16438268578793153
Epoch 3000, Error: 0.09044275402776122
Epoch 4000, Error: 0.06549239852846861
Epoch 5000, Error: 0.0529701612331762
Epoch 6000, Error: 0.04530890606234747
Epoch 7000, Error: 0.04006626247773649
Epoch 8000, Error: 0.03621378331048006
Epoch 9000, Error: 0.03324035673782845

Final predicted output:
[[0.01077995]
 [0.03407637]
 [0.03399428]
 [0.95539521]]