import numpy as np
class LinearRegression:
    def __init__(self):
        self.features=0
        self.targets=0
        #Learning rate help how fast gradient descent should work
        self.learning_rate=0
        self.epochs=0
        self.number_of_samples=0
        self.weights=0
        self.bias=0
        # print()
    def train(self,X,Y,learning_rate=0.0001,epochs=1000):
        #Setting learning_rate and epochs 
        self.features=X
        self.targets=Y
        self.learning_rate=learning_rate
        self.epochs=epochs
        # 
        # initializing the weights vector with zeros.Number of zeros depends on number of features
        self.weights=np.zeros(len(self.features[0]))
        self.number_of_samples=len(self.features)
        self.squared_mean_error()
        for _ in range(self.epochs):
            sum_of_gradient=np.zeros(len(self.features[0]))
            sum_of_bias=0
            #For each epoch we will iterate over each sample
            for j in range(len(self.weights)):
                #for each weight we will change the calculate gradient descent
                for i in range(self.number_of_samples):
                    #For each sample we will calculate the prediction
                    prediction=self.weights.dot(self.features[i])+self.bias
                    gradient=(prediction-self.targets[i])*self.features[i][j]
                    sum_of_gradient[j]+=gradient
            #calculate bias gradient
            for k in range(self.number_of_samples):
                prediction=self.weights.dot(self.features[k])+self.bias
                gradient=(prediction-self.targets[k])
                sum_of_bias+=gradient
            for k in range(len(self.features[0])):
                self.weights[k]-=(self.learning_rate*sum_of_gradient[k])/self.number_of_samples
            self.bias-=(learning_rate/self.number_of_samples)*sum_of_bias
            self.squared_mean_error()
            # print(f"After {_+1} epochs squared mean error is {self.squared_mean_error()}")

    def squared_mean_error(self):
        
        #Calculating the mean squared error  
        loss=0
        # print(self.weights.shape,self.features.shape)
        for i in range(self.number_of_samples):
            x_samples=self.features[i]
            y_samples=self.targets[i]
            # The line `prediction=np.dot(self.weights,x_samples)+self.bias` is calculating the
            # predicted value for a given input sample `x_samples` using the linear regression model.
            prediction=self.weights.dot(x_samples)+self.bias
            #calculating the squared error between the actual target value
            # `y_samples` and the predicted value `prediction` for each sample in the dataset.
            loss+=(y_samples-prediction)**2
        error=1/(2*self.number_of_samples)*loss
        print("After {0} iterations b = {1}, m = {2}, error = {3}".format(self.epochs, self.bias, self.weights, error))
        return error

if __name__=='__main__':
    # X=np.array([[1,2],[3,4],[4,5]])
    # Y=np.array([1,2,3])
    # model=LinearRegression()
    # model.train(X,Y,epochs=0)
    # print(model.squared_mean_error())
    # print(model.weights)
    # model.squared_mean_error()
    # model=LinearRegression(X,Y)
    # Load the CSV data
    data = np.loadtxt('data.csv', delimiter=',')

    # Split into features (X) and target (Y)
    X = data[:, 0]  # First column as X
    if X.ndim==1:
        X=X.reshape(-1,1)
    Y = data[:, 1]  # Second column as Y
    model=LinearRegression()
    model.train(X,Y,learning_rate=0.0001,epochs=10)
    