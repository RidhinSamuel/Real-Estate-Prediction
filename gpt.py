import numpy as np

class LinearRegression:
    def __init__(self):
        self.features = 0
        self.targets = 0
        self.learning_rate = 0
        self.epochs = 0
        self.number_of_samples = 0
        self.weights = 0
        self.bias = 0

    def train(self, X, Y, learning_rate=0.0001, epochs=1000):
        self.features = X
        self.targets = Y
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Ensure features is 2D, even if there's only one feature
        if self.features.ndim == 1:
            self.features = self.features.reshape(-1, 1)

        self.weights = np.zeros(self.features.shape[1])  # Number of weights equals the number of features
        self.number_of_samples = len(self.features)
        self.squared_mean_error()
        # Gradient Descent
        for _ in range(self.epochs):
            sum_of_gradient = np.zeros(len(self.weights))
            sum_of_bias = 0
            
            # Iterate over each sample
            for i in range(self.number_of_samples):
                prediction = self.weights.dot(self.features[i]) + self.bias
                error = prediction - self.targets[i]
                
                # Calculate gradients
                sum_of_bias += error
                for j in range(len(self.weights)):
                    sum_of_gradient[j] += error * self.features[i][j]

            # Update weights and bias
            self.weights -= (self.learning_rate * sum_of_gradient) / self.number_of_samples
            self.bias -= (self.learning_rate * sum_of_bias) / self.number_of_samples

            # Optionally print the error for debugging
            self.squared_mean_error()

    def squared_mean_error(self):
        loss = 0
        for i in range(self.number_of_samples):
            prediction = self.weights.dot(self.features[i]) + self.bias
            loss += (self.targets[i] - prediction) ** 2
        error = loss / (2 * self.number_of_samples)
        print("Current weights:", self.weights)
        print("Current bias:", self.bias)
        print("Current error:", error)
        return error

if __name__ == '__main__':
    # # Example with two features
    # X_multi = np.array([[1, 2], [3, 4], [4, 5]])
    # Y_multi = np.array([1, 2, 3])
    # model_multi = LinearRegression()
    # model_multi.train(X_multi, Y_multi, epochs=10)
    
    # # Example with one feature
    # X_single = np.array([1, 2, 3])
    # Y_single = np.array([1, 2, 3])
    # model_single = LinearRegression()
    # model_single.train(X_single, Y_single, epochs=10)
    # Load the CSV data
    data = np.loadtxt('data.csv', delimiter=',')

    # Split into features (X) and target (Y)
    X = data[:, 0]  # First column as X
    # if X.ndim==1:
    #     X=X.reshape(-1,1)
    Y = data[:, 1]  # Second column as Y
    model=LinearRegression()
    model.train(X,Y,learning_rate=0.0001,epochs=10)
