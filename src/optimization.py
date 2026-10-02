import numpy as np
import tensorflow as tf
import scipy.optimize as sopt

class OptimizationProblem():
    """
    Class representing an optimization problem.
    """
    def __init__(self, variables, loss_train, loss_valid, patience = None):
        """
        Parameters
        ----------
        variables
            List of trainable variables.
        loss_train
            Callable returning the loss function on the training dataset.
        loss_valid
            Callable returning the loss function on the validation dataset.
        patience
            Number of iterations without improvement of the validation loss
            after which the optimization stops (None: never stop early).
        """
        self.variables = variables if isinstance(variables, (list, tuple)) else [variables]
        # Keras 3 wraps each tf.Variable in its own class: the gradient tape watches the tf.Variable
        self.watched = [v if isinstance(v, tf.Variable) else v.value for v in self.variables]
        self.loss_train = loss_train
        self.loss_valid = loss_valid
        self.patience = patience
        self.stitcher = VariablesStitcher(self.variables)

        self.compile()

        self.iteration = 0
        self.iterations_history = list()
        self.loss_train_history = list()
        self.loss_valid_history = list()
        self.best_valid = np.inf
        self.best_iteration = 0
        self.best_params = None
        self.stop = False
        self.iteration_callback()

    def compile(self):
        self.ag_train_loss = tf.function(self.loss_train)
        self.ag_train_grad = tf.function(self.compute_gradient)
        self.ag_train_loss_grad = tf.function(lambda params: self.get_gradient_and_loss(params_1d = params))
        self.ag_valid_loss = tf.function(self.loss_valid)

        print('Tracing functions with autograph...')
        self.ag_train_loss()
        self.ag_train_grad()
        self.ag_train_loss_grad(self.stitcher.stitch(self.variables).numpy())
        self.ag_valid_loss()
        print('Tracing completed.')

    def get_gradient_and_loss(self, params_1d):
        self.stitcher.update_variables(params_1d)
        with tf.GradientTape(watch_accessed_variables = False) as tape:
            tape.watch(self.watched)
            loss_value = self.loss_train()
        grads = self.stitcher.stitch(tape.gradient(loss_value, self.watched, unconnected_gradients = tf.UnconnectedGradients.ZERO))
        return loss_value, grads

    def ag_train_loss_grad_numpy(self, params_1d):
        loss, grad = self.ag_train_loss_grad(params_1d)
        return loss.numpy(), grad.numpy()

    def compute_gradient(self):
        with tf.GradientTape(watch_accessed_variables = False) as tape:
            tape.watch(self.watched)
            loss_value = self.loss_train()
        return tape.gradient(loss_value, self.watched, unconnected_gradients = tf.UnconnectedGradients.ZERO)

    def iteration_callback(self):
        if self.iteration % 10 == 0:
            self.iterations_history.append(self.iteration)
            self.loss_train_history.append(self.ag_train_loss())
            self.loss_valid_history.append(self.ag_valid_loss())
            print('epoch% 5d   -   training loss: %1.3e   -   validation loss %1.3e' %
                  (self.iteration, self.loss_train_history[-1], self.loss_valid_history[-1]))
            # keep the weights at the minimum of the validation loss
            if self.loss_valid_history[-1] < self.best_valid:
                self.best_valid = float(self.loss_valid_history[-1])
                self.best_iteration = self.iteration
                self.best_params = self.stitcher.stitch(self.variables).numpy()
            elif self.patience is not None and self.iteration - self.best_iteration >= self.patience:
                self.stop = True
        self.iteration += 1

    def optimize_keras(self, num_epochs, optimizer):
        for _ in range(num_epochs):
            optimizer.apply_gradients(zip(self.ag_train_grad(), self.variables))
            self.iteration_callback()
            if self.stop:
                break

    def optimize_BFGS(self, num_epochs):
        # scipy's BFGS ends as soon as a line search fails: it is restarted from
        # the current point until the iterations are spent, the validation loss
        # stops improving, or a restart makes no progress.
        def callback(_):
            self.iteration_callback()
            if self.stop:
                raise StopIteration

        remaining = num_epochs
        while remaining > 0 and not self.stop:
            options = {'maxiter': remaining, 'gtol': 1e-100}
            init_params = self.stitcher.stitch(self.variables).numpy()
            start = self.iteration
            try:
                sopt.minimize(fun = self.ag_train_loss_grad_numpy,
                        x0 = init_params,
                        method = 'BFGS',
                        jac = True,
                        tol = 1e-100,
                        options = options,
                        callback = callback)
            except StopIteration:
                break
            progress = self.iteration - start
            remaining -= progress
            if progress <= 5:
                break
            if remaining > 0:
                print('BFGS restarted after %d iterations' % progress)

    def restore_best(self):
        """Set the variables to the weights at the minimum of the validation loss."""
        self.stitcher.update_variables(self.best_params)
        return self.best_valid

class VariablesStitcher:
    """
    Helper class to reshape a list of tf.Variable's into a 1D tf.Tensor/np.array and vice versa.
    """
    def __init__(self, variables):

        self.variables = variables

        # obtain the shapes of the variables
        self.shapes = tf.shape_n(self.variables)
        self.n_tensors = len(self.shapes)

        count = 0
        self.idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(self.shapes):
            n = np.prod(shape)
            self.idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        self.__num_variables = count

        self.part = tf.constant(part)

    @property
    def num_variables(self):
        return self.__num_variables

    def update_variables(self, params_1d):
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.variables[i].assign(tf.reshape(param, shape))

    def reverse_stitch(self, params_1d):
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        return [tf.reshape(param, shape) for i, (shape, param) in enumerate(zip(self.shapes, params))]

    def stitch(self, v = None):
        return tf.dynamic_stitch(self.idx, self.variables if v is None else v)
