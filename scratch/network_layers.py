import numpy as np
import cv2
import json
import random

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)

    def forwards(self, input):
        """Perform forward propagation through all layers."""
        for layer in self.layers:
            input = layer.forwards(input)
        return input

    def backwards(self, output_gradient, learning_rate):
        """Perform backward propagation through all layers."""
        for layer in reversed(self.layers):
            output_gradient = layer.backwards(output_gradient, learning_rate)

    def train(self, json_file_path, epochs, batch_size, learning_rate):
        """Train the model using the given data."""
        
        with open(json_file_path, 'r') as file:
            lines = file.readlines()

        for epoch in range(epochs):
            random.shuffle(lines)  # Shuffle the lines for each epoch

            for i in range(0, len(lines), batch_size):
                batch = lines[i:i + batch_size]

                # Initialize batch accumulators
                accumulated_gradients = [np.zeros_like(layer.weights) for layer in self.layers if hasattr(layer, 'weights')]
                accumulated_biases = [np.zeros_like(layer.biases) for layer in self.layers if hasattr(layer, 'biases')]

                actual_batch_size = 0  # In case some images fail to load

                for line in batch:
                    label_data = json.loads(line)
                    lanes = label_data['lanes']
                    h_samples = label_data['h_samples']
                    raw_file = label_data['raw_file']

                    image_path = f"archive/TUSimple/train_set/{raw_file}"
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Failed to load image: {image_path}")
                        continue
                    predictions = self.forwards(image)
                    loss_gradient = binary_cross_entropy_prime(np.array(lanes), predictions)

                    self.backwards(loss_gradient, learning_rate)

                    for j, layer in enumerate(self.layers):
                        if hasattr(layer, 'weights'):
                            accumulated_gradients[j] += layer.weights_gradient
                        if hasattr(layer, 'biases'):
                            accumulated_biases[j] += layer.biases_gradient

                
                    actual_batch_size += 1
                # Only update if something actually ran
                if actual_batch_size == 0:
                    continue

                # Gradient descent step: average and update
                for j, layer in enumerate(self.layers):
                    if hasattr(layer, 'weights'):
                        layer.weights -= learning_rate * (accumulated_gradients[j] / actual_batch_size)
                    if hasattr(layer, 'biases'):
                        layer.biases -= learning_rate * (accumulated_biases[j] / actual_batch_size)

            print(f"Epoch {epoch + 1}/{epochs} completed")

class Convolution():
    def __init__(self, input_shape, kernel_size, kernel_count):
        # input shape = (height, width, depth (color channels))
        # kernel count = int
        # output shape = height, width, depth (kernel_count)
        # kernel shape = height, width = height, color channels, kernel count



        input_height, input_width, input_depth = input_shape
        self.kernel_count = kernel_count
        self.input_shape = input_shape
        self.input_depth = input_depth
        
        # output shape for a valid cross-correlation, where only positions where the kernel fully
        # fits inside the input matrix are used
        self.output_shape = (input_height - kernel_size + 1, input_width - kernel_size + 1, kernel_count)        
        
        
        # this personally does not make much sense. The depth of the image does not increase the number
        # of kernels. The depth of the image just increases the depth of each kernel. I think.
        self.kernels_shape = (kernel_size, kernel_size, input_depth, kernel_count)

        # there is no variable for the bias shape since it is equal to the output shape


        # initialize the Kernels and biases
        self.weights = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        # print("bias shape: ", self.biases.shape)


    def forwards(self, input):
        print("Performing Convolution forwards")
        # print("input shape for cross correlation: ", input.shape) # 720, 1280, 3
        self.input = input # create instance variable "input" which is a single image with some depth maybe
        self.output = np.copy(self.biases) # since the bias matrix is simply added, it makes sense to initialize the output variable to it and then add everything else
        for i in range(self.kernel_count):
            for j in range(self.input_depth):
                # print(self.input[:, :, j].shape)
                # print("output_shape: ", self.output.shape)
                # print("input_deDSpth: ", self.input_depth)
                # print("self.kernels shape", self.kernels.shape)
                self.output[:, :, i] += self.valid2DCrossCorrelation(self.input[:, :, j], self.weights[:, :, j, i])
        return self.output


    def backwards(self, output_gradient): # output gradient is dE/dy
        print("Performing Convolution backwards")
        self.weights_gradient = np.zeros(self.kernels_shape)
        self.biases_gradient = np.zeros(self.kernel_count)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.kernel_count):
            for j in range(self.input_depth):
                self.weights_gradient[:, :, i, j] += self.valid2DCrossCorrelation(self.input[:, :, j], output_gradient[i])
                input_gradient[:, :, j] += self.full2DConvolution(output_gradient[i], self.weights[:, :, i, j])

            self.biases_gradient[i] = np.sum(output_gradient[i])
        return input_gradient


    def valid2DCrossCorrelation(self, inputMatrix, kernelMatrix):
        # print("Correlation starts")
        input_height, input_width = inputMatrix.shape
        kernel_height, kernel_width = kernelMatrix.shape
        # print("input matrix: ", inputMatrix.shape)
        # print("kernel matrix: ", kernelMatrix.shape)
        outputMatrixWidth = input_width - kernel_width + 1
        outputMatrixHeight = input_height - kernel_height + 1
        outputMatrix = np.empty((outputMatrixHeight, outputMatrixWidth))
        for i in range (outputMatrixHeight):
            for j in range (outputMatrixWidth):
                outputMatrix[i, j] = np.sum(inputMatrix[i:i+kernel_height, j:j+kernel_width] * kernelMatrix)
        # print("correlation_output: ", outputMatrix.shape)
        return outputMatrix
    

    def full2DConvolution(self, inputMatrix, kernelMatrix):
        input_height, input_width = inputMatrix.shape
        kernel_height, kernel_width = kernelMatrix.shape

        # Rotate kernel 180 degrees
        rotated_kernel = np.flip(kernelMatrix)

        # Pad input matrix
        pad_height = kernel_height - 1
        pad_width = kernel_width - 1
        padded_input = np.pad(inputMatrix, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
        # print ("Padded input: ", padded_input)
        # Output shape
        output_height = input_height + kernel_height - 1
        output_width = input_width + kernel_width - 1
        outputMatrix = np.zeros((output_height, output_width), dtype=float)

        # Perform convolution
        for i in range(output_height):
            for j in range(output_width):
                patch = padded_input[i:i+kernel_height, j:j+kernel_width]
                outputMatrix[i, j] = np.sum(patch * rotated_kernel)
                # print ("outputMatrix. i: ", i, " j: ", j, " \n", outputMatrix, "\n")


        return outputMatrix

    
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


class ReLU:
    def forwards(self, input):
        print("Performing ReLU forwards")
        # print("input to ReLU: ", input.shape)
        self.input = input
        return np.maximum(0, input)

    def backwards(self, output_gradient, learning_rate):
        print("Performing ReLU backwards")
        return output_gradient * (self.input > 0)

class Flatten:
    def forwards(self, input):
        self.input_shape = input.shape
        return input.flatten().reshape(1, -1)
    
    # backward just brings the flattened image back to its unflattened form
    def backwards(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)
    
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(1, output_size)

    def forwards(self, input):
        self.input = input
        print("input dense shape: ", input.shape)
        return np.dot(input, self.weights) + self.biases
        
    def backwards(self, output_gradient, learning_rate):
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Gradient descent update
        self.biases_gradient = output_gradient
        
        return input_gradient # pass this to the previous layer
    



class Reshape():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forwards(self, input):
        return np.reshape(input, self.output_shape)
    
    def backwards(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

class Softmax:
    def forwards(self, input):
        self.output = np.exp(input - np.max(input, axis=1, keepdims=True))  # For numerical stability
        self.output /= np.sum(self.output, axis=1, keepdims=True)
        return self.output

    def backwards(self, output_gradient, learning_rate=None):
        # Create an empty array for the input gradients
        input_gradient = np.empty_like(output_gradient)

        for i, (single_output, single_output_grad) in enumerate(zip(self.output, output_gradient)):
            # Flatten column vectors
            single_output = single_output.reshape(-1, 1)
            # Jacobian matrix of the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Apply chain rule
            input_gradient[i] = np.dot(jacobian_matrix, single_output_grad)

        return input_gradient


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forwards(self, input):
        self.input = input
        h, w, c = input.shape  # (height, width, channels)
        out_h = (h - self.size) // self.stride + 1
        out_w = (w - self.size) // self.stride + 1
        output = np.zeros((out_h, out_w, c))

        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    region = input[
                        i * self.stride : i * self.stride + self.size,
                        j * self.stride : j * self.stride + self.size,
                        ch
                    ]
                    output[i, j, ch] = np.max(region)
        return output

    def backwards(self, output_gradient, learning_rate=None):
        input_gradient = np.zeros_like(self.input)
        h, w, c = self.input.shape
        out_h = (h - self.size) // self.stride + 1
        out_w = (w - self.size) // self.stride + 1

        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    region = self.input[
                        i * self.stride : i * self.stride + self.size,
                        j * self.stride : j * self.stride + self.size,
                        ch
                    ]
                    max_val = np.max(region)
                    for m in range(self.size):
                        for n in range(self.size):
                            if region[m, n] == max_val:
                                input_gradient[
                                    i * self.stride + m,
                                    j * self.stride + n,
                                    ch
                                ] += output_gradient[i, j, ch]
                                break  # only the first max gets the gradient
        return input_gradient

