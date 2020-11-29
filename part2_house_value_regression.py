import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pickle
import math
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, ClassifierMixin
import random
# from skopt.searchcv import BayesSearchCV
# from skopt.space import Integer, Real, Categorical 
# from skopt.utils import use_named_args
# from skopt import gp_minimize


class Regressor(BaseEstimator, ClassifierMixin):

    def __init__(self, x, nb_epoch = 1000, batch_size =  64, learning_rate = 0.01, H1 = 45, H2 = 30, H3 = 15 , DRP = 0.1):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """
        
        #Attributes to store constants to be applied on test data
        self.yScaler = preprocessing.RobustScaler()
        self.xScaler = preprocessing.RobustScaler()
        self.lb = preprocessing.LabelBinarizer()
        
        self.x = x
        if x is not None:
            X, _ = self._preprocessor(x, training = True) 

        self.loss_values = []
        self.input_size = X.shape[1]
        self.output_size = 1
        
        self.nb_epoch = nb_epoch 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.DRP = DRP

        
        self.net = Net(self.input_size, self.H1, self.H2, self.H3, self.output_size, self.DRP)

        return

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """
             
        #Solve SettingWithCopyWarning
        _x = x.copy()
        if y is not None:
            _y = y.copy()
            
        
        #Normalize y, save scaler for inverse transform
        if y is not None:
            column = ['median_house_value']
            if training:
                _y.loc[:,column] = self.yScaler.fit_transform(_y.loc[:,column])
            else:
                _y.loc[:,column] = self.yScaler.transform(_y.loc[:,column])


        #Normalize x
        columnsToNormalize = ['longitude', 'latitude', 'housing_median_age', 
                              'total_rooms', 'total_bedrooms', 'population', 
                              'households', 'median_income']
        
        if training:
            _x.loc[:,columnsToNormalize] = self.xScaler.fit_transform(_x.loc[:,columnsToNormalize])
        else:
            _x.loc[:,columnsToNormalize] = self.xScaler.transform(_x.loc[:,columnsToNormalize])
        

        #Handle missing values
        _x = _x.fillna(x.mean())
        

        #Handle textual values
        oceanProximityValues = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

        if training:
            self.lb.fit(oceanProximityValues)

        ohe = pd.DataFrame(self.lb.transform(_x.loc[:,'ocean_proximity']), columns=self.lb.classes_)
        _x = _x.drop(columns=['ocean_proximity'])
        _x.reset_index(drop=True, inplace=True)
        _x = _x.join(ohe)

        return _x.values, (_y.values if isinstance(y, pd.DataFrame) else None) 
    
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        #get the torch of X and Y
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        self.net.train()  # set training mode
        self.loss_values = []
        
        #loss_func = nn.L1Loss() 
        
        # mean squared error
        loss_func = nn.MSELoss() 
        #Adam optimiser
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate) 
        #number of itterations per epoch
        
        
        print("Starting training with parameters: epoch: {}, batch size: {}, learning rate: {}"
              .format(self.nb_epoch, self.batch_size, self.learning_rate))
        print("Net Architecture: \n{}".format(self.net))
        #number of possible values to take
        n_items = len(X)

        itt_per_epoch = n_items // self.batch_size
        #iterate through number of epochs
        for epoch in range(1, self.nb_epoch+1):
            #reshuffling
            X, Y = sklearn.utils.shuffle(X, Y)
            X_t_full = T.tensor(X).float()
            Y_t_full = T.tensor(Y).float()
            
            #loss per epoch
            running_loss = 0.0
            #itterate thorugh number of itterations per epoch
            for i in range(itt_per_epoch):
                #get indices to the values for the current batch
                #do not replace so no repeats
                current_batch = np.random.choice(n_items, self.batch_size, 
                                                 replace=False)
                #get the X and Y tensors
                X_t = X_t_full[current_batch]
                Y_t = Y_t_full[current_batch].view(self.batch_size,1)
                #set gradient of optimised Tensors to 0
                optimizer.zero_grad()
                #get the prediction
                y_pred = self.net(X_t)
                #calculate the loss
                loss_obj = loss_func(y_pred, Y_t)
                #dloss/dx
                loss_obj.backward()
                #single optimisation step
                optimizer.step()
                #add current loss to the overall loss per epoch
                running_loss += loss_obj.item()
            
                #for each 100th epoch, print the epoch's loss outcome
                if (epoch % 100 == 0 and (i+1) % itt_per_epoch== 0) or (epoch == 1 and (i+1) % itt_per_epoch== 0):
                    print('epoch: %d, epoch loss: %.3f' %
                          (epoch, running_loss ))
            #add the final loss for this epoch
            self.loss_values.append(running_loss)
         
        #plotting the overall loss epoch graph for training    
        plt.plot(list(range(1, self.nb_epoch+1)),self.loss_values)
        plt.title("Model's Loss")
        plt.ylabel('Loss')
        plt.xlabel('Epoch Number')
        print("Training complete \n")
        #return model
        return self

    
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """
        #get the tensor for the X value
        X, _ = self._preprocessor(x, training = False) 
        X = T.tensor(X).float()
        #start the evaluation mode
        self.net.eval()
        #reduces memory usage and speed up computations 
        with T.no_grad():
            #predict the y 
            y_pred = self.net(X)
            
        trueOutput = self.yScaler.inverse_transform(y_pred)
        return trueOutput

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #get the Y tensor of true values
        _, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        #make it into numpy
        #y_true = Y.detach().numpy()
        #get the prediction using the predict method
        y_pred = self.predict(x)

        #calculate the mse for the values
        mse = mean_squared_error(y, y_pred)
        #square root to get rmse
        rmse = math.sqrt(mse)
        return rmse

class Net(nn.Module):
  def __init__(self, D_in, H1, H2, H3, D_out, DRP):
    #initiate the Net
    super(Net, self).__init__()
    #3FFC layers and 1 output layer
    self.inpt = nn.Linear(D_in, H1)
    self.hid1 = nn.Linear(H1, H2)
    self.hid2 = nn.Linear(H2, H3)
    self.oupt = nn.Linear(H3, D_out)
    #dropout (so not fully connected)
    self.drop = nn.Dropout(DRP, inplace=True)

  def forward(self, x):
    #relu activation function
    z = F.relu(self.inpt(x))
    #applying dropout after FFC layer
    z = self.drop(z)
    #relu activation function
    z = F.relu(self.hid1(z))
    #applying dropout after FFC layer
    z = self.drop(z)
    #relu activation function
    z = F.relu(self.hid2(z))
     #applying dropout after FFC layer
    z = self.drop(z)
    #output
    z = self.oupt(z)
    return z
    
def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x, y): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #parameters we want to search
    params = [{'nb_epoch': range(400, 1000),
                'batch_size': [32, 64, 96], 
                'learning_rate': [0.002, 0.001, 0.0005],
                'H1': range(30,70), 
                'H2': range(15, 40),
                'H3': range(2, 30),
                'DRP': [0, 0.1, 0.15]
                }]

    #setting up the search
    search = RandomizedSearchCV(
        Regressor(x),
        param_distributions=params,
        cv = 3,
        n_iter=1,
        scoring="neg_mean_squared_error",
        )
    #fitting the search wiht the parameters
    search.fit(x, y)
    
    print('Best Parameters: \n{}'.format(search.best_params_))

    return  search.best_estimator_


def example_main():

    output_label = "median_house_value"

    # reading in csv data file
    data = pd.read_csv("housing.csv") 

    # Spliting input and output

    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    test_split = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_split)

    # Training
    best_regressor = RegressorHyperParameterSearch(x_train, y_train)
    regressor = best_regressor
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

if __name__ == "__main__":
    example_main()



