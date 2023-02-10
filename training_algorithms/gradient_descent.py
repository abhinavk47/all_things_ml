import numpy as np
import random
import time
import math
import argparse
parser = argparse.ArgumentParser()


class GradientDescent:
    def __init__(self, learning_rate, tolarance, l2_parameter, batch_size):
        """
        Parameters:
            learning_rate (float): parameter to control the rate at which weights, bias are updated
            tolarance (float): termination parameter
            l2_parameter (float): l2 regularization panality parameter
            batch_size : size of each batch
        """
        self.learning_rate = learning_rate
        self.tolarance = tolarance
        self.l2_parameter = l2_parameter
        self.batch_size = batch_size

    def create_mini_batches(self, feature_values, outputs):
        """
        Creates a list of batches of feature values and outputs

        Parameters:
            feature_values (list of arrays): feature values
            outputs (list): original outputs
        
        Returns:
            list: batches of feature values, outputs
        """
        mini_batches = []
        batch_len = math.ceil(len(feature_values)/self.batch_size)
        for batch_idx in range(batch_len):
            mini_batch = [feature_values[batch_idx*self.batch_size:batch_idx*self.batch_size+self.batch_size],
                            outputs[batch_idx*self.batch_size:batch_idx*self.batch_size+self.batch_size]]
            mini_batches.append(mini_batch)
        return mini_batches

    def update_learning_rate(self, new_rate_of_change, old_rate_of_change):
        """
        Updates learning_rate based on rate of change of rate of change
        """
        if (new_rate_of_change < 1000 and new_rate_of_change - old_rate_of_change > 100) or \
                (new_rate_of_change < 100 and new_rate_of_change - old_rate_of_change > 10) or \
                (new_rate_of_change < 10 and new_rate_of_change - old_rate_of_change > 1):
            self.learning_rate *= 0.9
            print("Decreased learning rate by 10%, current learning rate", self.learning_rate)

    def gradient_descent(self, feature_values, original_outputs, weights, bias):
        """
        Updates weights, bias using gradient descent

        Parameters:
            feature_values (list of arrays): feature values
            original_outputs (list): original outputs
            weights (list): initial defined weights
            bias (float): initial defined bias

        Returns:
            list: updated weights
            int: updated bias
        """
        n = len(feature_values)
        # Making copies of weights, bias in case the learning rate is too high to converge
        weights_copy = weights.copy()
        bias_copy = bias
        # Run gradiant descent to update weights, bias
        mse_old = sum([(np.dot(np.transpose(weights), x) + bias - y)**2 for x, y in zip(feature_values, original_outputs)])/n
        iteration_num = 0
        old_rate_of_change = None
        start = time.time()
        while True:
            # Using MSE loss function sum(((y - (w*x + b))**2)/N)
            if self.batch_size:
                mini_batches = create_mini_batches(feature_values, original_outputs, self.batch_size)
                for mini_batch in mini_batches:
                    feature_values_batch, outputs_batch = mini_batch
                    num_features = len(feature_values_batch[0])
                    w_gradient = [0]*num_features
                    for feature_idx in range(num_features):
                        w_gradient[feature_idx] = 2*sum([(np.dot(np.transpose(weights), x) + bias - y)*x[feature_idx] for x, y in zip(feature_values_batch, outputs_batch)])/n
                        if self.l2_parameter:
                            w_gradient[feature_idx] += self.l2_parameter*(weights[feature_idx]**2)
                    b_gradient = 2*sum([np.dot(np.transpose(weights), x) + bias - y for x, y in zip(feature_values_batch, outputs_batch)])/n
                    # Update weights, bias
                    for feature_idx in range(num_features):
                        weights[feature_idx] -= self.learning_rate*w_gradient[feature_idx]
                    bias -= self.learning_rate*b_gradient
            else:
                num_features = len(feature_values[0])
                w_gradient = [0]*num_features
                for feature_idx in range(num_features):
                    w_gradient[feature_idx] = 2*sum([(np.dot(np.transpose(weights), x) + bias - y)*x[feature_idx] for x, y in zip(feature_values, original_outputs)])/n
                    if self.l2_parameter:
                        w_gradient[feature_idx] += self.l2_parameter*(weights[feature_idx]**2)
                b_gradient = 2*sum([np.dot(np.transpose(weights), x) + bias - y for x, y in zip(feature_values, original_outputs)])/n
                # Update weights, bias
                for feature_idx in range(num_features):
                    weights[feature_idx] -= self.learning_rate*w_gradient[feature_idx]
                bias -= self.learning_rate*b_gradient
            # Calculate the loss
            mse_new = sum([(np.dot(np.transpose(weights), x) + bias - y)**2 for x, y in zip(feature_values, original_outputs)])/n
            new_rate_of_change = abs(mse_new-mse_old)
            if old_rate_of_change is not None and new_rate_of_change > old_rate_of_change:
                print("Learning rate", self.learning_rate, "is too high, decreasing by 1%", "time taken", time.time()-start)
                start = time.time()
                iteration_num = 0
                self.learning_rate *= 0.99
                weights = weights_copy.copy()
                bias = bias_copy
                old_rate_of_change = None
                mse_old = mse_new
                continue
            # Checking for convergence
            if new_rate_of_change < self.tolarance:
                print("iteration", iteration_num, "Learning rate", self.learning_rate, "Final Loss", mse_new)
                break
            self.update_learning_rate(new_rate_of_change, old_rate_of_change)
            iteration_num += 1
            if iteration_num%1000 == 0:
                print("iteration", iteration_num, "Time taken", time.time()-start, "Last rate of change", abs(mse_new-mse_old))
            old_rate_of_change = new_rate_of_change
            mse_old = mse_new
        
        return weights, bias

def real_function(x, weights, bias):
    """
    Calculates the real output using a linear function
    
    Parameters:
        x (list): feature values
        weights (list): original weights
        bias (float): original bias
    
    Returns:
        int: output value
    """
    # Convert to numpy array
    if type(x) == list:
        x = np.array(x)
    if type(weights) == list:
        weights = np.array(weights)
    weights = np.transpose(weights)

    output = np.dot(x, weights) + bias
    return output

def main(learning_rate, tolarance, l2_parameter=None, batch_size=None):
    training_obj = GradientDescent(learning_rate, tolarance, l2_parameter, batch_size)
    # Define original weights and bias
    num_features = 5
    original_weights = [random.uniform(-2, 10) for _ in range(num_features)]
    original_bias = random.randint(0, 100)
    # Define training set
    n = 1000
    x = [np.array([random.uniform(1, 1000) for _ in range(num_features)]) for _ in range(n)]
    y = []
    for feature_values in x:
        output = real_function(feature_values, original_weights, original_bias)
        y.append(output)
    # Normalise features (Scaling to unit length)
    divisors = []
    for feature_idx in range(num_features):
        divisor = math.sqrt(sum([feature_values[feature_idx]**2 for feature_values in x]))
        for i in range(n):
            x[i][feature_idx] /= divisor
        divisors.append(divisor)
    # Define initial values of weight and bias
    w = [0]*num_features
    b = 0
    # Run gradient descent to get the model parameters
    w, b = training_obj.gradient_descent(x, y, w, b)
    print("Original weights, bias:", original_weights, original_bias)
    print("Model's weights, bias:", w, b)
    print("Corrected model's weights, bias:", [weight/divisor for weight, divisor in zip(w, divisors)], b)

if __name__ == "__main__":
    parser.add_argument("--learning_rate", help="learning rate for gradient descent")
    parser.add_argument("--tolarance", help="tolarance to rate of change of the lose function")
    parser.add_argument("--batch_size", help="batch size during stochastic gradient descent")
    parser.add_argument("--l2_parameter", help="l2 regularization parameter")
    args = parser.parse_args()
    learning_rate = float(args.learning_rate)
    tolarance = float(args.tolarance)
    l2_parameter = args.l2_parameter
    if l2_parameter:
        l2_parameter = float(l2_parameter)
    batch_size = args.batch_size
    if batch_size:
        batch_size = int(batch_size)
    main(learning_rate, tolarance, l2_parameter, batch_size)
