import scipy.io
import numpy as np
import matplotlib.pyplot as plt  

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Sigmoid Activation Function
def sigmoid(x):
    return 1. / (1. + np.exp(x))

# Fully Connected Neural Network
class NeuralNetwork():
    def __init__(self, A1, A2, B1, B2):
        self.A1 = A1
        self.A2 = A2
        self.B1 = B1
        self.B2 = B2

    def forward(self, Z):
        # W1 = A1 ∗ Z + B1
        W1 = np.dot(self.A1, Z) + self.B1
        # Z1 = max{W1, 0} (ReLU)
        Z1 = relu(W1)
        # W2 = A2 ∗ Z1 + B2
        W2 = np.dot(self.A2, Z1) + self.B2
        # X = 1./[1 + exp(W2)](Sigmoid)
        X = sigmoid(W2)
        # Return output
        return X

if __name__ == "__main__":
    # Load the generative model
    mat_data = scipy.io.loadmat('data/data1.mat')

    # Access the data
    A1 = mat_data['A_1']  # Matrix A1 128 × 10
    A2 = mat_data['A_2']  # Matrix A2 784 × 128
    B1 = mat_data['B_1']  # Vector B1 128 × 1
    B2 = mat_data['B_2']  # Vector B2 784 × 1

    # Initialize the Neural Network
    nn = NeuralNetwork(A1, A2, B1, B2)

    # Generate 100 realizations of Z and apply the generator
    generated_images = []
    for _ in range(100):
        Z = np.random.normal(0, 1, (10, 1))  # Input Z ~ N(0,1) size 10 × 1
        X = nn.forward(Z)  # Generate the output size 784x1
        X_2D = X.reshape(28, 28).T  # Reshape and transpose for correct orientation
        generated_images.append(X_2D) # Append Image

    # Create a 10 × 10 grid of images
    grid_image = np.zeros((28 * 10, 28 * 10))
    for i in range(10):
        for j in range(10):
            grid_image[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = generated_images[i * 10 + j]

    # Display images on 10 × 10 grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_image, cmap='gray', origin='upper')
    plt.title("Generated 8s")
    plt.axis('off')
    plt.show()