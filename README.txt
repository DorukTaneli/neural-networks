FILES IN THE DIRECTORY:
part1_nn_lib.py:    		    code for part1
iris.dat:           		    data for part1
part2_house_value_regression.py:    code for part2
housing.csv:                        data for part2
part2_model.pickle:                 resulting model from part2
analysis.py:                        produces histograms of housing.csv columns
graphs/*:                           histograms as png for each column
report.pdf:                         architecture, methodology and explanations of part2


TO RUN CODE FOR PART1:
>python3 part1_nn_lib.py

DEFAULT: the example_main() method is called which trains a neural network for classification on the iris dataset (iris.dat) and prints the train loss, validation loss and accuracy

OPTIONAL: To train a neural network on another classification or regression problem, adapt the code in the example.main() method or create and call an alternative main method. In the main method:
- Initialise the network using the MultiLayerNetwork class and specify the input dimension, number of neurons per layer and activation functions per layer
- Load the dataset, specify features x and labels y, split into train and validation set
- Initialise the trainer using the Trainer class and specify network, batch-size, epochs, learning rate, loss function & shuffle flag
- preprocessor.apply(): preprocess the data using the Preprocessor class (optional)
- trainer.train(): train the network on the training data
- trainer.eval(): evaluate the predictions on the validation data 



TO RUN CODE FOR PART2:
>python3 part2_house_value_regression.py
-Finds the best hyperparameters using randomizedSearchCV, prints the best hyperparameters.
-Trains the model using 80% of housing.csv using the best hyperparameters.
-Saves the model in part2_model.pickle.
-Tests the model using remaining 20% of housing.csv, prints root mean squared error.


EXTRAS:
>python3 analysis.py
-Produces histogram for each column in housing.csv, saves them in /graphs/ folder


For detailed explanations, check report.pdf
