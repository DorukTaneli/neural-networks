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

    def __init__(self, x, nb_epoch = 1000, batch_size =  64, learning_rate = 0.01, H1 = 60, H2 = 20, H3 = 10 , DRP = 0.15):
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

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        #Attributes to store constants to be applied on test data
        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit(x["ocean_proximity"]) 
        self.scaler1 = preprocessing.RobustScaler()
        self.scaler1.fit(x[['longitude', 'latitude', 'housing_median_age', 
                            'total_rooms', 'total_bedrooms', 'population', 
                            'households', 'median_income']])
        
        # Replace this code with your own
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

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

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

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
             
        if 'ocean_proximity' in x.columns: #not preprocessed           
            # Handle textual values:
            x = x.join(pd.DataFrame(self.lb.transform(x["ocean_proximity"]), columns=self.lb.classes_))
            x = x.drop(['ocean_proximity'], axis=1)
            
            # Handle missing values:
            x = x.fillna(x.mean()); #replaces missing values with mean
            
            # Normalize
            x[['longitude', 'latitude', 'housing_median_age', 
                'total_rooms', 'total_bedrooms', 'population', 
                'households', 'median_income']] = self.scaler1.transform(x[['longitude', 'latitude', 'housing_median_age', 
                                                                            'total_rooms', 'total_bedrooms', 'population', 
                                                                            'households', 'median_income']])
            
        #print("\preprocessing data:")
        #print(x)
        #x.info(verbose=True)
        
        print(type(x.values))
        # Return preprocessed x and y, return None for y if it was None
        #return T.tensor(x.values).float(), (T.tensor(y.values).float() if isinstance(y, pd.DataFrame) else None)
        return x.values, (y.values if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
 
    
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
        
        #loss_func = nn.L1Loss() 
        
        # mean squared error
        loss_func = nn.MSELoss() 
        #Adam optimiser
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate) 
        #number of itterations per epoch
        itt_per_epoch = self.nb_epoch // self.batch_size
        
        print("Starting training with parrameters: epoch: {}, batch size: {}, learning rate: {}"
              .format(self.nb_epoch, self.batch_size, self.learning_rate))
        print("Net Architecture: \n{}".format(self.net))
        #number of possible values to take
        n_items = len(X)
        #initiate the list of loss values
        
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
            
                #for each 100th epoch and 10th batch print the loss outcome
                if epoch % 100 == 0 and (i+1) % 10== 0:
                    print('epoch: %d, batch: %d, loss: %.3f' %
                          (epoch, i + 1,loss_obj.item() ))
            #add the final loss for this epoch
            self.loss_values.append(running_loss)
         
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
            y_pred = self.net(X).detach().numpy()
        return y_pred

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
        #plotting the overall loss epoch graph for training    
        plt.plot(list(range(1, self.nb_epoch+1)),self.loss_values)
        plt.title("Params: epoch: {}, batch size {},learning rate {}".format(self.nb_epoch, self.batch_size, self.learning_rate))
        plt.ylabel('Loss')
        plt.xlabel('Epoch Number')

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
    self.hid1 = nn.Linear(D_in, H1)
    self.hid2 = nn.Linear(H1, H2)
    self.hid3 = nn.Linear(H2, H3)
    self.oupt = nn.Linear(H3, D_out)
    #dropout (so not fully connected)
    self.drop = nn.Dropout(DRP, inplace=True)

  def forward(self, x):
    #relu activation function
    z = F.relu(self.hid1(x))
    #applying dropout after FFC layer
    z = self.drop(z)
    #relu activation function
    z = F.relu(self.hid2(z))
    #applying dropout after FFC layer
    z = self.drop(z)
    #relu activation function
    z = F.relu(self.hid3(z))
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
    params = [{'nb_epoch': range(200, 1000),
               'batch_size': [32, 64, 96], 
               'learning_rate': [0.002, 0.001, 0.0005],
               'H1': range(40,80), 
               'H2': range(20, 60),
               'H3': range(2, 40),
               'DRP': [0, 0.1, 0.2]
               }]
    #initiating the model
    model = Regressor(x)
    #setting up the search
    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_jobs=1,
        cv=5,
        n_iter=30,
        scoring="neg_mean_squared_error",
        verbose=4,
        random_state=42
        )
    #fitting the search wiht the parameters
    search.fit(x, y)
    
    search.get_params(deep=True)
    print(search.best_params_)

    return  search.best_params_


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
    #params = RegressorHyperParameterSearch(x_train, y_train)
    regressor = Regressor(x_train, nb_epoch = 1000, batch_size =  64, learning_rate = 0.001, H1 = 42, H2 = 27, H3 = 10, DRP = 0.1)
    regressor.fit(x_test, y_test)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()


