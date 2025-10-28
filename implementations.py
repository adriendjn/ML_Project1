import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N,), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        loss: loss value (scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    loss = 1/2 * np.mean(e**2)
    return loss

def compute_mse_gradient(y, tx, w):
    """Computes the gradient of the MSE loss function at w.

    Args:
        y: numpy array of shape=(N,), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        grad: numpy array of shape (D,) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    grad = -(1/len(e)) * tx.T @ e
    return grad

def compute_mse_stoch_gradient(y, tx, w):
    """Compute the stochastic gradient of the MSE loss function at w, from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B,), B is the number of samples in the batch.
        tx: numpy array of shape=(B,D), D is the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        grad: numpy array of shape (D,) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y - tx @ w
    grad = -(1/len(e)) * tx.T @ e
    return grad

def sigmoid(t):
    t = np.clip(t, -500, 500)
    return 1.0/(1 + np.exp(-t))

def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a mini-batch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx').
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the mini-batches.

    Args:
        y: numpy array of shape=(N,), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        batch_size: optional, scalar denoting the number of element in each batch, default: 1.
        num_batches: optional, scalar denoting the number of batches to yield, default: 1.
        shuffle: optional, bool indicating if data should be shuffled before sampling the mini-batches, default: True.

    Yields:
        Iterator over **num_batches** mini-batches, each of size **batch_size**, on the data of **y** and **tx**.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Implement Linear regression using the Gradient Descent (GD) algorithm and MSE loss.

    Args:
        y: numpy array of shape=(N,), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters.
        max_iters: scalar denoting the total number of iterations of GD.
        gamma: scalar denoting the stepsize.

    Returns:
        w: model parameters as numpy array of shape (D,), for the last iteration of GD.
        loss: loss value (scalar) for the last iteration of GD.
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_mse(y, tx, w)

    for n_iter in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_mse(y, tx, w)

    return (w, loss)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Implement Linear regression using the Stochastic Gradient Descent (SGD) algorithm and MSE loss.

    Args:
        y: numpy array of shape=(N,), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters.
        max_iters: scalar denoting the total number of iterations of SGD.
        gamma: scalar denoting the stepsize.

    Returns:
        w: model parameters as numpy arrays of shape (D,), for the last iteration of SGD.
        loss: loss value (scalar) for the last iteration of SGD.
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_mse(y, tx, w)

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx):
            grad = compute_mse_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
            loss = compute_mse(minibatch_y, minibatch_tx, w)

    return (w, loss)

def least_squares(y, tx):
    """Implement Least Squares regression using normal equations and MSE loss.
       returns optimal weights and loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: loss value (scalar) for the Least Squares regression.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # returns mse, and optimal weights
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_mse(y, tx, w)
    return (w, loss)

def ridge_regression(y, tx, lambda_):
    """Implement Ridge regression using normal equations and MSE loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization parameter.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: loss value (scalar) for the Ridge regression.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N, D = tx.shape
    I = np.eye(D)
    w = np.linalg.solve(tx.T @ tx + lambda_ * 2 * N * I, tx.T @ y)
    loss = compute_mse(y, tx, w)
    return (w, loss) 

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression using the Gradient Descent (GD) algorithm and Log loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters.
        max_iters: scalar denoting the total number of iterations of SGD.
        gamma: scalar denoting the stepsize.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: loss value (scalar) for the logistic regression.
    """
    w = initial_w
    pred = sigmoid(tx @ w)
    loss_t = y.T @ np.log(pred) + (1 - y).T @ np.log(1 - pred)
    loss = np.squeeze(-loss_t) * (1 / y.shape[0])

    for n_iter in range(max_iters):
        pred = sigmoid(tx @ w)
        grad = tx.T @ (pred - y) * (1 / y.shape[0])
        w -= gamma * grad
        pred = sigmoid(tx @ w)
        loss_t = y.T @ np.log(pred) + (1 - y).T @ np.log(1 - pred)
        loss = np.squeeze(-loss_t) * (1 / y.shape[0])
    return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """implement regularized logistic regression using the Gradient Descent (GD) algorithm and Log loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization parameter.
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters.
        max_iters: scalar denoting the total number of iterations of SGD.
        gamma: scalar denoting the stepsize.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: loss value (scalar) for the logistic regression.
    """
    y_binary = (y + 1) / 2
    w = initial_w.copy()
    losses = []
    
    for n_iter in range(max_iters):
        z = tx @ w
        pred = sigmoid(z)
        pred = np.clip(pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_binary * np.log(pred) + (1 - y_binary) * np.log(1 - pred)) + lambda_ * np.sum(w**2)
        losses.append(loss)
        grad = tx.T @ (pred - y_binary) / len(y) + 2 * lambda_ * w
        w -= gamma * grad
        if n_iter == max_iters - 1:
            print(f"Final loss: {loss:.4f}")
    
    return w, losses

