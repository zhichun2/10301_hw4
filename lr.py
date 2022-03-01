import numpy as np

# setting global variables
assert(len(sys.argv) == 8)
formatted_train_input = sys.argv[1]
formatted_validation_input = sys.argv[2]
formatted_test_input = sys.argv[3]
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics_out = sys.argv[6]
num_epoch = sys.argv[7]
learning_rate = sys.argv[8]
def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

# organize a tsv file to an array of tuples (x, y)
def formatted2arr(path):
    res = np.empty()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            label, vector = row[0], row[1:]
            tmp = np.array((vector, label))
            res = np.vstack(res, tmp)
    return res

# returns dj/dthetaj
def differentiate(x, y, theta, i):
    return (y[i] - sigmoid(np.dot(theta, x))) # is theta transpose x equal to dot?

def train(theta, X, y, num_epoch, learning_rate):
    # TODO: Implement `train` using vectorization
    # too many for loops?
    for k in range num_epoch:
        for i in range np.shape(X)[0]:
            theta = theta + learning_rate * np.dot((differentiate(X[i], y, theta, i), X[i]))
    return theta

def predict(theta, X):
    # TODO: Implement `predict` using vectorization
    prediction = np.matmul(X, np.transpose(theta))
    label = np.zeros(np.shape(predictions), 1)
    for i in prediction:
        if prediction[i][0] >= 0.5:
            label[i][0] = 1
    return label

def compute_error(y_pred, y):
    # TODO: Implement `compute_error` using vectorization
    res = np.transpose(y_pred) - np.transpose(y)
    error = 0
    for i in res:
        if res[i] != 0:
            error += 1
    return error/np.shape(y)

def lr(formatted_train_input, formatted_validation_input, formatted_test_input, train_out, test_out, metrics_out, num_epoch, learning_rate):
    train_arr = np.loadtxt(formatted_train_input, delimiter='\t')
    train_theta = np.zeros(np.shape(train_arr)[1]) # first col is intercept
    train_X = np.hstack((np.ones((np.shape(train_arr)[0], 1))), train_arr[1:])
    train_y = train_arr[:, 0]
    train_theta = train(train_theta, train_X, train_y, num_epoch, learning_rate)
    train_prediction = predict(train_theta, train_X)
    np.savetxt(train_out, train_prediction, delimiter='\t', fmt'%d')
    train_error = compute_error(train_prediction, train_y)
    
    test_arr = np.loadtxt(formatted_test_input, delimiter='\t')
    test_X = np.hstack((np.ones((np.shape(test_arr)[0], 1))), test_arr[1:])
    test_y = test_arr[:, 0]
    test_prediction = predict(train_theta, test_X)
    np.savetxt(test_out, test_prediction, delimiter='\t', fmt'%d')
    test_error = compute_error(test_prediction, test_y)

    # format for matrix_out?
    with open(metrics_out, 'w') as f_out:
        f_out.write('training error: ' + str(train_error) + '\n')
        f_out.write('test error: ' + str(test_error) + '\n')

if __name__ == '__main__':
    lr(formatted_train_input, formatted_validation_input, formatted_test_input, train_out, test_out, metrics_out, num_epoch, learning_rate)