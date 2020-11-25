import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
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
        self.lb.fit_transform(x["ocean_proximity"]) 
        self.scaler1 = preprocessing.RobustScaler()
        self.scaler1.fit_transform(x[['longitude', 'latitude', 'housing_median_age', 
                                      'total_rooms', 'total_bedrooms', 'population', 
                                      'households', 'median_income']])
        
        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.net = Net()

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
             
        if training: #training data: calculate and apply preprocessing values            
            #Handle textual values:
            #fit and transform
            x = x.join(pd.DataFrame(self.lb.transform(x["ocean_proximity"]), columns=self.lb.classes_))
            x = x.drop(['ocean_proximity'], axis=1)
            
            ###Handle missing values:
            x = x.fillna(x.mean()); #replaces missing values with mean
            
            #normalize
            x[['longitude', 'latitude', 'housing_median_age', 
                'total_rooms', 'total_bedrooms', 'population', 
                'households', 'median_income']] = self.scaler1.transform(x[['longitude', 'latitude', 'housing_median_age', 
                                                                            'total_rooms', 'total_bedrooms', 'population', 
                                                                            'households', 'median_income']])
            
            #print("\ntraining data:")
            #print(x)
            #x.info(verbose=True)
            
            
        else: #test data: apply existing values
            #Handle textual values:
            #only transform
            x = x.join(pd.DataFrame(self.lb.transform(x["ocean_proximity"]), columns=self.lb.classes_))
            x = x.drop(['ocean_proximity'], axis=1)
            
            ###Handle missing values:
            x = x.fillna(x.mean()); #replaces missing values with mean
            
            #normalize:
            x[['longitude', 'latitude', 'housing_median_age', 
                'total_rooms', 'total_bedrooms', 'population', 
                'households', 'median_income']] = self.scaler1.transform(x[['longitude', 'latitude', 'housing_median_age', 
                                                                            'total_rooms', 'total_bedrooms', 'population', 
                                                                            'households', 'median_income']])
            
            #print("\ntest data:")
            #print(x)
            #x.info(verbose=True)
        
    
        
        # Return preprocessed x and y, return None for y if it was None
        return torch.tensor(x.values), (torch.tensor(y.values) if isinstance(y, pd.DataFrame) else None)

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

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        self.net.train()  # set training mode
        print(self.net)  # net architecture
        bat_size = 64
        loss_func = nn.MSELoss()  # mean squared error
        optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        n_items = len(X)
        batches_per_epoch = n_items // bat_size
        max_batches = self.nb_epoch * batches_per_epoch
        print("Starting training")
        for b in range(max_batches):
            curr_bat = np.random.choice(n_items, bat_size,
                                        replace=False)
            X_t = X[curr_bat].float()
            Y_t = Y[curr_bat].view(bat_size,1).float()
            optimizer.zero_grad()
            oupt = self.net(X_t)
            loss_obj = loss_func(oupt, Y_t)
            loss_obj.backward()
            optimizer.step()
            if b % (max_batches // 10) == 0:
                print("batch = %6d" % b, end="")
                print("  batch loss = %7.4f" % loss_obj.item(), end="\n")
                self.net.eval()
                self.net.train()
    
        print("Training complete \n")
        return self.net

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
       
    
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """
        
        

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        X=X.float()
        return self.net(X)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

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

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        _, Y = self._preprocessor(x, y = y, training = False) # Do not forget
    
        y_true = Y.float().detach().numpy()
        y_pred = self.predict(x).detach().numpy()

        
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        
        return rmse

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()
#     self.hid1 = nn.Linear(13, 10)  # 13-(10-10)-1
#     self.hid2 = nn.Linear(10, 10)
#     self.oupt = nn.Linear(10, 1)

#   def forward(self, x):
#     z = nn.ReLU(self.hid1(x))
#     z = torch.tanh(self.hid2(z))
#     z = self.oupt(z)  # no activation, aka Identity()
#     return z
    
# class Net(torch.nn.Module):
#     def __init__(self, n_feature = 13, n_hidden = 10, n_output =1) :
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
#         self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = self.predict(x)             # linear output
#         return x
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = nn.Linear(13, 32)  # 13-(10-10)-1
    self.hid2 = nn.Linear(32, 16)
    self.oupt = nn.Linear(16, 1)

  def forward(self, x):
    z = torch.relu(self.hid1(x))
    z = torch.relu(self.hid2(z))
    z = self.oupt(z)  # no activation, aka Identity()
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



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

