import numpy as np
import pickle


def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current = x
        sigmoid_output = 1/(1+np.exp(-x))
        
        return sigmoid_output 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        derivation_sigmoid = (1/(1+np.exp(-self._cache_current)))*(1-(1/(1+np.exp(-self._cache_current))))
        grad_X_sigmoid = grad_z*derivation_sigmoid
        return grad_X_sigmoid
        #Irgendetwas stimmt hier glaube ich noch nicht...

        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current = x
        relu_output = np.maximum(x, 0)
        return relu_output
        

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        relu = np.maximum(self._cache_current, 0)
        derivation_relu = np.minimum(1, relu)
        grad_X_relu = grad_z*derivation_relu
        
        return grad_X_relu

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        #self._W = None
        #self._b = None
        
        #use the imput and output size as well as the xavier_init method to set initial W and b.
        self._W = xavier_init([n_in, n_out], gain = 1.0)
        #print("initial weights", self._W)
        self._b = xavier_init(n_out, gain = 1.0)
        #print("initial bias", self._b)

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
      
        weights_incl_bias = self._W.copy()
        weights_incl_bias = np.append(weights_incl_bias, [self._b], axis = 0)
        #print("FORWARD: Weights incl bias",np.shape(weights_incl_bias ))
        ones = np.ones(np.shape(x)[0])
        
        x_incl_one = x.copy()
        x_incl_one =  np.concatenate((x_incl_one, np.transpose([ones])), axis = 1)
        #print("FORWARD: X incl ones",x_incl_one)
        output = np.matmul(x_incl_one, weights_incl_bias)
        #print("Output should be 2x4", output)
        
        self._cache_current = x
        
        return output
        
        
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._grad_W_current = np.matmul(np.transpose(self._cache_current), grad_z)
        #print("gradient weights", self._grad_W_current)
        ones = np.ones(np.shape(grad_z)[0])
        self._grad_b_current = np.matmul(ones, grad_z)
        #print("shape current weights", np.shape(self._W))
        grad_X_current = np.matmul(grad_z, np.transpose(self._W))
        #print("gradient X", self._grad_W_current)
        return grad_X_current 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W = self._W - learning_rate*self._grad_W_current
        self._b = self._b - learning_rate*self._grad_b_current

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        #Create empty list to store the layers
        layers=[]
        dimensions = [input_dim] + neurons
        #print("Input_dimension", input_dim)
        #print("Neurons", neurons)
        #print("Concat Dimensions", dimensions)
        
        for dim in range(1, len(dimensions)):
            layers.append(LinearLayer(dimensions[dim-1], dimensions[dim]))
            if (activations[dim-1] == "relu"):
                layers.append(ReluLayer())
            if (activations[dim-1] == "sigmoid"):
                layers.append(SigmoidLayer())
        
        #print(layers)
        self._layers = layers
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        for layer in range(0, len(self._layers)):
           x = self._layers[layer].forward(x)
           #print("Shape of Forward pass:",np.shape(x))
        return x
        
        

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- batch_size, number of final neurons).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in range(len(self._layers)-1, -1, -1):
           #print("Shape of grad_z", np.shape(grad_z))
           grad_z = self._layers[layer].backward(grad_z)
           #print("Shape of Gradient", grad_z)
        
        return grad_z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in range(0, len(self._layers)):
            if (isinstance(self._layers[layer], LinearLayer)):
                #print("It's a Linear layer!")
                self._layers[layer].update_params(learning_rate)
 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if (loss_fun == "mse"):
            self._loss_layer = MSELossLayer() #potentially needs a layer as input
        
        if (loss_fun == "cross_entropy"):
            self._loss_layer = CrossEntropyLossLayer()
            
        else: print("Please chose a valid loss function")
        self.loss_collector = np.zeros(nb_epoch)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        shuffable = np.concatenate((input_dataset, target_dataset), axis=1)
        #print(shuffable)
        np.random.shuffle(shuffable)
        #print(shuffable)
        input_columns=np.shape(input_dataset)[1]
        #print(input_columns)
        input_dataset = shuffable[:, :input_columns]
        target_dataset = shuffable[:, input_columns:]
        #print(target_dataset)
        
        return input_dataset, target_dataset
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        
        
        for epoch in range(0, self.nb_epoch):
        
        
            if (self.shuffle_flag):
                Trainer.shuffle(input_dataset, target_dataset) #shuffle is static, this is fine
            
        
            input_dataset_batches = np.array_split(input_dataset, self.batch_size)
            #print("Batch size:",np.shape(input_dataset_batches))
            target_dataset_batches = np.array_split(target_dataset, self.batch_size)
            
         
            
            for x in range(np.shape(input_dataset_batches)[0]):
                
                
                #print("Batch Iteration (", batch, "/", np.shape(input_dataset_batches)[0],")" )
                prediction = self.network.forward(input_dataset_batches[x])
                loss = self._loss_layer.forward(prediction, target_dataset_batches[x])
                
                loss_gradient = self._loss_layer.backward()
                self.network.backward(loss_gradient)
                self.network.update_params(self.learning_rate)
            
            #print("Epoch", epoch)
            #print("Loss:", loss)
            self.loss_collector[epoch]=loss
    

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        prediction = self.network.forward(input_dataset)
        loss = self._loss_layer.forward(prediction, target_dataset)
        return loss

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
            
        
        self.min = data.min(axis=1)
        self.max = data.max(axis=1)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        #Min max normalisation to range [-1, 1]
        
        normalized_data = data.copy()
        
        for x in range(len(data)):
            normalized_data[x] = -1 + ((data[x]-self.min[x])*(2))/(self.max[x]-self.min[x])
        
        return normalized_data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        reverted_data = data.copy()
        
        for x in range(len(data)):   
            reverted_data[x] = ((data[x]+1)*(self.max[x]-self.min[x]))/2 + self.min[x]
        return reverted_data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    #print(np.shape(dat))
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    #print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
   # print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    #print("Validation accuracy: {}".format(accuracy))
    
    
    ###################################
    #MY TESTS
    ###################################
    
    import matplotlib.pyplot as plt
    plt.plot(trainer.loss_collector)
    
    '''
    #initialisation
    print(xavier_init([3, 4]))
    
    x=np.array([[1,2,3],[4,5, 6]]) #batch size 2, input size 3
    print(x.max(axis=1))
    print(x-3)
    
    #Test the linear layer
    ones = np.ones(np.shape(x)[0])
    print(ones)
    x_incl_one = x.copy()
    print(x_incl_one)
    x_incl_one = np.concatenate((x_incl_one, np.transpose([ones])), axis = 1)
    print(x_incl_one)
    layer = LinearLayer(n_in=3, n_out=4)
    print(layer.forward(x))
    
    
    grad_loss_wrt_outputs = np.ones((2, 4))
    
    #grad_loss_wrt_inputs = layer.backward(grad_loss_wrt_outputs) 
    #print(grad_loss_wrt_inputs)
    #print(layer.update_params(0.5))
    
    #Test the sigmoid layer
    print("SIGMOID LAYER TESTS")
    test_z = np.ones((2, 3))*0.5
    
    #print(1/(1-2**x))
    sigmoid=SigmoidLayer()
    print("Forward:")
    print(sigmoid.forward(x))
    print("Backward:")
    print(sigmoid.backward(test_z))
    
    
     #Test the relu layer
    print("RELU LAYER TESTS")
    relu_x=np.array([[1,2,3],[4,-5, 6]])
    #relu_test_2 = np.maximum(relu_test, 0)
    #print(np.minimum(1, relu_test_2))
    relu=ReluLayer()
    print("Forward:")
    print(relu.forward(relu_x))
    print("Backward:")
    print(relu.backward(test_z))
    
    
    network = MultiLayerNetwork(input_dim=3, neurons=[6, 2], activations=["relu", "sigmoid"])
    network.forward(x)
    network.backward(np.ones((2, 2)))
    network.update_params(0.02)
    
    
    #Test training
    input_dataset = np.array([[1,2,3],[3,4,5],[1,2,3],[3,4,5],[1,2,3],[3,4,5],[1,2,3],[3,4,5]])
    target_dataset = np.array([[1],[3],[1],[3],[1],[3],[1],[3]])
    shuffable = np.concatenate((input_dataset, target_dataset), axis=1)
    print(shuffable)
    np.random.shuffle(shuffable)
    print(shuffable)
    input_columns=np.shape(input_dataset)[1]
    print(input_columns)
    input_dataset = shuffable[:, :input_columns]
    output_dataset = shuffable[:, input_columns:]
    print(output_dataset)
    
    
    print("TRAINING")
    trainer = Trainer( network=network, batch_size=2, nb_epoch=10, learning_rate=1.0e-3, shuffle_flag=True, loss_fun="mse")
    trainer.train(input_dataset, target_dataset)
    
    print("Validation loss = ", trainer.eval_loss(input_dataset, target_dataset))
    '''
     ###################################
    #END MY TESTS
    ###################################


if __name__ == "__main__":
    example_main()
