FILES IN THE DIRECTORY:
part1_nn_lib.py:    code for part1
iris.dat:           data for part1
part2_house_value_regression.py:    code for part2
housing.csv:                        data for part2
part2_model.pickle:                 resulting model from part2
analysis.py:                        produces histograms of housing.csv columns
graphs/*:                           histograms as png for each column
report.pdf:                         architecture, methodology and explanations of part2


TO RUN CODE FOR PART1:



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
