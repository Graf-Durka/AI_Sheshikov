import numpy as np

class FC():
    def __init__(self, input_size, output_size):
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        return np.dot(self.input, self.weights) + self.biases
    
    def backward(self, output_gradient):
        grad_input = np.dot(output_gradient, self.weights.T)

        gard_weights = np.dot(self.input.T, output_gradient)
        grad_biases = np.sum(output_gradient, axis=0, keepdims=True)

        return grad_input, gard_weights, grad_biases

class ReLU():
    def forward(self, x):
        self.input = x
        return np.maximum(0, self.input)
    
    def backward(self, output_gradient):
        return output_gradient * (self.input > 0)
    
# Не стал добавлять паддинг и страйд, оставил базовыми 0 и 1 соответственно
class Conv():
    def __init__(self, kernel_size, in_channels, out_channels):
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        scale = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.kernel = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * scale

    def forward(self, x):
        # Если первая свёртка, добавляем измерение канала
        if len(x.shape) == 2:
            self.input = np.expand_dims(x, axis=2)
        else:
            self.input = x
            
        h, w, _ = self.input.shape
        self.output = np.zeros((h - self.kernel_size + 1, w - self.kernel_size + 1, self.out_channels))

        for oc in range(self.out_channels):
            for i in range(h - self.kernel_size + 1):
                for j in range(w - self.kernel_size + 1):
                    region = self.input[i:i+self.kernel_size, j:j+self.kernel_size, :]
                    self.output[i, j, oc] = np.sum(region * self.kernel[:, :, :, oc])
        return self.output
    
    def backward(self, output_gradient):
        grad_input = np.zeros_like(self.input)
        grad_kernel = np.zeros_like(self.kernel)

        h, w, _ = self.input.shape
        for oc in range(self.out_channels):
            for i in range(h - self.kernel_size + 1):
                for j in range(w - self.kernel_size + 1):
                    region = self.input[i:i+self.kernel_size, j:j+self.kernel_size, :]
                    grad_kernel[:, :, :, oc] += region * output_gradient[i, j, oc]
                    grad_input[i:i+self.kernel_size, j:j+self.kernel_size, :] += self.kernel[:, :, :, oc] * output_gradient[i, j, oc]

        # Возвращаем grad_input в исходном формате (если он был 2D)
        if grad_input.shape[2] == 1:
            grad_input = grad_input.squeeze(axis=2)
            
        return grad_input, grad_kernel
    
class MaxPool2d():
    def __init__(self, kernel_size, channels):
        self.kernel_size = kernel_size
        self.channels = channels

    def forward(self, x):
        self.input = x
        h, w, _ = self.input.shape
        out_h = h // self.kernel_size
        out_w = w // self.kernel_size
        self.output = np.zeros((out_h, out_w, self.channels))

        for c in range(self.channels):
            for i in range(out_h):
                for j in range(out_w):
                    i_start = i * self.kernel_size
                    j_start = j * self.kernel_size
                    region = self.input[i_start:i_start+self.kernel_size, j_start:j_start+self.kernel_size]
                    self.output[i, j, c] = np.max(region)
        return self.output

    def backward(self, output_gradient):
        grad_input = np.zeros_like(self.input)

        h, w, _ = self.input.shape
        out_h = h // self.kernel_size
        out_w = w // self.kernel_size

        for c in range(self.channels):
            for i in range(out_h):
                for j in range(out_w):
                    i_start = i * self.kernel_size
                    j_start = j * self.kernel_size
                    region = self.input[i_start:i_start+self.kernel_size, j_start:j_start+self.kernel_size]
                    max_val = np.max(region)
                    grad_input[i_start:i_start+self.kernel_size, j_start:j_start+self.kernel_size] += (region == max_val) * output_gradient[i, j, c]

        return grad_input
    
class Flatten():
    def forward(self, x):
        self.input = x
        return self.input.flatten().reshape(1, -1)

    def backward(self, output_gradient):
        return output_gradient.reshape(self.input.shape)
    
class Softmax():
    def forward(self, x):
        self.input = x
        exp_x = np.exp(self.input - np.max(self.input))
        return exp_x / np.sum(exp_x)

    def backward(self, output_gradient):
        s = self.forward(self.input).reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        return np.dot(jacobian, output_gradient.T).T
    
# Реализовал стандартный ADAM, beta1 и beta2 как в torch
class ADAM():
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i in range(len(self.parameters)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.parameters[i] -= update

class SimpleCNN():

    # Был вариант со скрытым полносвязным слоем перед выходным слоем + ReLU,
    # Но решил оставить исключительно свёрточную архитектуру
    def __init__(self):
        # 1 слой свертки и пуллинга: Вход 28x28x1 -> Выход 13x13x8
        self.conv1 = Conv(kernel_size=3, in_channels=1, out_channels=8)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, channels=8)
        
        # 2 слой свертки: Вход 13x13x8 -> Выход 12x12x16
        self.conv2 = Conv(kernel_size=3, in_channels=8, out_channels=16)
        self.relu2 = ReLU()
        
        # Слой пулинга: Вход 11x11x16 -> Выход 5x5x16
        self.pool2 = MaxPool2d(kernel_size=2, channels=16)
        self.flatten = Flatten()
        
        # Полносвязный слой 5 * 5 * 16 = 400
        self.fc1 = FC(input_size=400, output_size=10)
        self.softmax = Softmax()

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        return self.softmax.forward(x)
    
    def backward(self, output_gradient):
        grad = self.softmax.backward(output_gradient)
        grad, grad_fc1_weights, grad_fc1_biases = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad, grad_conv2_kernel = self.conv2.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad, grad_conv1_kernel = self.conv1.backward(grad)

        return [grad_conv1_kernel, grad_conv2_kernel, grad_fc1_weights, grad_fc1_biases]
    
    def step(self, grads, optimizer):
        optimizer.step(grads)

    def save_weights(self, filepath):
        np.savez(filepath, 
                 conv1_kernel=self.conv1.kernel, 
                 conv2_kernel=self.conv2.kernel,
                 fc1_weights=self.fc1.weights, 
                 fc1_biases=self.fc1.biases)

    def load_weights(self, filepath):
        data = np.load(filepath)
        self.conv1.kernel = data['conv1_kernel']
        self.conv2.kernel = data['conv2_kernel']
        self.fc1.weights = data['fc1_weights']
        self.fc1.biases = data['fc1_biases']

class CrossEntropyLoss():
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return -np.sum(targets * np.log(predictions + 1e-8))

    def backward(self):
        return -self.targets / (self.predictions + 1e-8)

# Скажу сразу, функцию писал не я
def fetch_mnist():
    import urllib.request
    import gzip
    import os
    
    url_base = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    
    for file in files:
        if not os.path.exists(file):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(url_base + file, file)
    
    with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
        
    with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
        
    return (x_train, y_train), (x_test, y_test)

def main():
    (x_train, y_train), (x_test, y_test) = fetch_mnist()
    
    train_size = 60000
    x_train = x_train[:train_size] / 255.0
    y_train = y_train[:train_size]

    # x_train = x_train / 255.0

    model = SimpleCNN()

    optimizer = ADAM(parameters=
    [
        model.conv1.kernel, 
        model.conv2.kernel,
        model.fc1.weights, 
        model.fc1.biases
    ], 
    
    learning_rate=0.001)
    loss_fn = CrossEntropyLoss()

    epochs = 3
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for i in range(len(x_train)):
            input_data = x_train[i]
            target = np.zeros((1, 10))
            target[0, y_train[i]] = 1

            predictions = model.forward(input_data)
            loss = loss_fn.forward(predictions, target)
            total_loss += loss
            
            if np.argmax(predictions) == y_train[i]:
                correct += 1

            output_gradient = loss_fn.backward()
            grads = model.backward(output_gradient)
            model.step(grads, optimizer)

        avg_loss = total_loss / len(x_train)
        accuracy = correct / len(x_train)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

    model.save_weights("simple_cnn_weights.npz")
    print("Weights saved to simple_cnn_weights.npz")

if __name__ == "__main__":
    main()