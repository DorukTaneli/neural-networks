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
        
        #Store x for the backward pass
        self._cache_current = x
        
        #Apply the sigmoid function to all values in x
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
        
        #Calculate the derivative of the activation function
        derivation_sigmoid = (1/(1+np.exp(-self._cache_current)))*(1-(1/(1+np.exp(-self._cache_current))))
        
        #Compute the gradients of loss w.r.t. the parameters & inputs of the layer
        grad_X_sigmoid = grad_z*derivation_sigmoid
        return grad_X_sigmoid
        
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
        
        
        
        #Apply the relu function to all values in x
        relu_output = np.maximum(x, 0)
        
        #Store the calculated output for the backward pass    
        self._cache_current = relu_output
        
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
        
        
        relu = self._cache_current
        
        #Calculate the derivative of the activation function
        relu[relu<=0] = 0
        relu[relu>0] = 1
 
        #Compute the gradients of loss w.r.t. the parameters & inputs of the layer
        grad_X_relu = grad_z*relu
        
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
        
        
        #Use the imput and output size as well as the xavier_init method to set initial W and b.
        self._W = xavier_init([n_in, n_out], gain = 1.0)
        self._b = xavier_init(n_out, gain = 1.0)
       

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
        
        #Concatenate weights and bias
        weights_incl_bias = self._W.copy()
        weights_incl_bias = np.append(weights_incl_bias, [self._b], axis = 0)
        
        #Add a column of ones to the input data to account for the bias term
        ones = np.ones(np.shape(x)[0])
        x_incl_one = x.copy()
        x_incl_one =  np.concatenate((x_incl_one, np.transpose([ones])), axis = 1)
        
        #Calculate the output by mltiplying the input data and the weights + bias
        output = np.matmul(x_incl_one, weights_incl_bias)
        
        #Store input data for the backward pass
        self._cache_current = x
        
        return output

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
        
        #Calculate & store the gradients w.r.t. the weights
        self._grad_W_current = np.matmul(np.transpose(self._cache_current), grad_z)
    
        #Calculate & store the gradients w.r.t. the bias
        ones = np.ones(np.shape(grad_z)[0])
        self._grad_b_current = np.matmul(ones, grad_z)
 
        #Calculate & return the gradients w.r.t. the input data
        grad_X_current = np.matmul(grad_z, np.transpose(self._W))
       
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
        
        #Perform one step of gradient descent to update and store the weights and bias
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
        
        #Combine the input dimension and layer dimensions in one array 
        dimensions = [input_dim] + neurons
        
        #Loop through the list of dimensions to create the layers accordingly
        for dim in range(1, len(dimensions)):
            #Create a linear layer (current dimension = output dimension; previous dimension = input dimension)
            layers.append(LinearLayer(dimensions[dim-1], dimensions[dim]))
            #Add an activation function layer in accordance with the activations list
            if (activations[dim-1] == "relu"):
                layers.append(ReluLayer())
            if (activations[dim-1] == "sigmoid"):
                layers.append(SigmoidLayer())
        
        #Store the generated list of layers
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
        
        #Perform a forward pass by looping through all layers
        for layer in range(0, len(self._layers)):
           x = self._layers[layer].forward(x)
          
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
        
        #Perform a backward pass by looping through all layers (back to front)
        for layer in range(len(self._layers)-1, -1, -1):
           grad_z = self._layers[layer].backward(grad_z)
          
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
        
        #Update the weights and biases for all linear layers
        for layer in range(0, len(self._layers)):
            if (isinstance(self._layers[layer], LinearLayer)):
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
        
        #Specify the loss function layer in accordance with the function parameters
        if (loss_fun == "mse"):
            self._loss_layer = MSELossLayer()
        
        elif (loss_fun == "cross_entropy"):
            self._loss_layer = CrossEntropyLossLayer()
         
        #Print an error message if the argument does not match the expected values
        else: print("Please choose a valid loss function - '"+loss_fun+"' is not supported.")
        
        #Create a vector to store loss for visualisation of the loss curve
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
        
        #Concatenate the input dataset and the target dataset to ensure the dataset is shuffled but the inputs and targets still match correctly
        if target_dataset.ndim == 1:
            to_squeeze = True
            shuffable = np.concatenate((input_dataset, np.transpose([target_dataset])), axis=1)
        else:
            to_squeeze = False
            shuffable = np.concatenate((input_dataset, target_dataset), axis=1)
        
        #Shuffle the data
        np.random.shuffle(shuffable)
        
        #Split shuffled input dataset and target dataset
        input_columns=np.shape(input_dataset)[1]
        input_dataset = shuffable[:, :input_columns]
        target_dataset = shuffable[:, input_columns:]

        #Restore original shape (if target is a vector)
        if to_squeeze:
            target_dataset = np.squeeze(target_dataset)
        
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
        
        #Loop through the specified number of epochs
        for epoch in range(0, self.nb_epoch):
        
            #Shuffle data if shuffle_flag = True
            if (self.shuffle_flag):
                input_dataset, target_dataset = Trainer.shuffle(input_dataset, target_dataset)
            
            #Split data into batches
            input_dataset_batches = np.array_split(input_dataset, self.batch_size)
            target_dataset_batches = np.array_split(target_dataset, self.batch_size)
            
            #Loop through the batches to train the neural network
            for x in range(np.shape(input_dataset_batches)[0]): 
                
                #Forward pass
                prediction = self.network.forward(input_dataset_batches[x])
                
                #Calculate loss
                loss = self._loss_layer.forward(prediction, target_dataset_batches[x])
                
                #Calculate gradient of loss
                loss_gradient = self._loss_layer.backward()
                
                #Backward pass
                self.network.backward(loss_gradient)
                
                #Update parameters
                self.network.update_params(self.learning_rate)
            
            #Store loss per epoch for visualisation of the loss curve
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
        
        #Use the (trained) network to produce a prediction
        prediction = self.network.forward(input_dataset)
        
        #Calculate and return the achieved loss
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
            
        #Store the mimimum and maximum values per column / feature
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

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
        
        #Min max normalisation – range [0, 1]
        
        normalized_data = data.copy()
        
        #Loop through the columns and normalise each on a scale of 0 to 1
        for x in range(np.shape(data)[1]):
            if (np.count_nonzero(data[:, x]) != 0):
                normalized_data[:, x] = (data[:, x]-self.min[x])/(self.max[x]-self.min[x])
        
        #Replace NaN with 0.
        normalized_data = np.nan_to_num(normalized_data)
        
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
        
        #Loop through the columns of the normalised data and restore the original values
        for x in range(np.shape(data)[1]):  
            reverted_data[:, x] = ((data[:, x])*(self.max[x]-self.min[x])) + self.min[x]
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
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))
    
    
    '''
    ###################################
    #ADDITIONAL TESTS
    ###################################
    
    import matplotlib.pyplot as plt
    plt.plot(trainer.loss_collector)
  
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
    
    
    
    #Test preprocessing
    data = np.array([[1.0,7.3,6.],[0.1,-1.,6.],[4.,5.,6.]])
    #data = np.array([[1,0,0],[0,0,0],[0.,0.,0.]])
    print("Original Data HOORZ")
    print(data)
    normalized_data= data.copy()
    mini = data.min( axis=0)
    maxi = data.max( axis=0)
    print("MIN pro Spalte", mini)
    print("MAX pro Spalte", maxi)
    #print(data[:,1])
    #print(data[:, 1]-mini[1])
    
    for x in range(np.shape(data)[1]):
        if (np.count_nonzero != 0):
            normalized_data[:, x] = -1 + ((data[:, x]-mini[x])*(2))/(maxi[x]-mini[x])
       
       
    print("Normalized")
    print(normalized_data)
    
    
    reverted_data = normalized_data.copy()
    for x in range(np.shape(normalized_data)[1]):   
            reverted_data[:, x] = ((normalized_data[:,x]+1)*(maxi[x]-mini[x]))/2 + mini[x]
            
    reverted_data = np.nan_to_num(reverted_data)
    
    print("Reverted", reverted_data)
    
    
    relu_test = data.copy()
    relu_test[relu_test<=0] = 0
    relu_test[relu_test>0] = 1
    print(relu_test)
    
    #relu = np.maximum(data, 0)
    #derivation_relu = np.minimum(1, relu)
    #grad_X_relu = grad_z*derivation_relu

    #print(grad_X_relu)   
   
    
    ###################################
    #END ADDITIONAL TESTS
    ###################################
    '''

if __name__ == "__main__":
    example_main()
