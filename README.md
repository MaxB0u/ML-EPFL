# ML-EPFL
Solutions to Machine Learning Projects at EPFL\
Link to the [dataset and submission platform](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).\
Register for this using your **EPFL email address** so that we can create a team for our submissions.

# Project Structure
- `src` directory contains all files related to the source code.
- `dataset` directory contains the dataset.
- `test` directory contains the code for testing / verifying correctness or accuracy.
- `docs` directory contains everything related to writing documentation for our work.
- `setup.sh` is the script to use for installing the dependencies into the machine which will run this project. (Supports installing dependencies for linux environments only).

# File Descriptions
**dataset/train.csv** - Training set of 250000 events. The file starts with the `ID` column, then the label column (the y you have to predict), and finally 30 feature columns.\
**dataset/test.csv** - The test set of around 568238 events - Everything as above, except the label is missing.\
**dataset/sample-submission.csv** - a sample submission file in the correct format. The sample submission always predicts -1, that is `background`.

For detailed information on the semantics of the features, labels, and weights, see the technical documentation from the LAL website on the task. Note that here for the EPFL course, we use a simpler evaluation metric instead (classification error).

#### Some details about the datasets:
- All variables are floating point, except PRI_jet_num which is integer
- Variables prefixed with `PRI` (for PRImitives) are “raw” quantities about the bunch collision as measured by the detector.
- Variables prefixed with `DER` (for DERived) are quantities computed from the primitive features, which were selected by the physicists of `ATLAS`.
- It can happen that for some entries some variables are meaningless or cannot be computed; in this case, their value is −999.0, which is outside the normal range of all variables.`

# Pipeline Summary

Here are the main steps that are executed when the run.py script is launched.
## Training
- Loading data from the train.csv file
- Data preprocessing
- Grid-search for the best hyperparameters with kfold cross validation for each hyperparameter combination. Note that in our final submission, the best hyperparameters have already been determined and this step is skipped
- Using the best hyperparameters, retrain the model over the whole dataset

## Testing
- Loading data from the test.csv file
- Data preprocessing
- Label prediction
- Write restuls to the submission.csv file in the /dataset folder

# Preprocessing

Here is a description of the preprocessing steps that gave us the best accuracy. A more detailed discussion on other preprocessing steps that were tried can be found in our report.

- Feature selection: Only the DER features are used from the dataset. The PRI features were highly correlated with them and gave little new predictive information to our models. The training was faster and the results slightly better when the PRI features were dropped.
- Replacing invalid values: Each invalid value (-999.0) is replaced by the median value of its corresponding feature
- Clip outliers: Outliers are defined as being values that fall further than 3 standard deviations from the mean of their feature (assuming a Gaussian distribution). The value of these outliers is replaced by the mean $+/-$ 3 standard deviations.
- Standardize: Transform the data so each feature has a mean of zero and a standard deviation of one.
- Feature expansion: Polynomial feature expansion of x. The resulting vector is of shape $(N,D*d)$ where $d$ is the degree of the polynomial expansion
- Add offset term: A new feature of all ones is prepended to x as an offset term in our models.

# Models

Here is a list of the models that were used for the project. The best model was selected from its loss within its own model class (linear and logistic regression) and using Accuracy and F1 score between different model classes. Since different model classes use different loss functions, their loss cannot be compared directly.

## Linear Regression

- Mean squared error gradient descent: Linear regression with Mean Squared Error (MSE) using the gradient descent algorithm
- Mean squared error stochastic gradient descent: Linear regression with Mean Squared Error (MSE) using the stochastic gradient descent algorithm
- Least squares: Linear regression using the normal equations (closed-form solutions)
- Ridge regression: Linear regression using the normal equations (closed-form solutions) as well as ridge regression (L2)

## Logistic Regression

- Logistic regression: Logistic regression using the gradient descent algorithm or Newton's method.
- Logistic regression with regularization: Logistic regression using the gradient descent algorithm with L2 regularization.

# Training and Validation

The test data is loaded from the train.csv dataset and is preprocessed with the steps mentionned in the Preprocessing section.

During our experimentations, the training was searching for the best hyperparameters to use in our final model.For each model, hyperparameters included gamma, lambda, the maximum number of iterations for gradient descent, the degree of the polynomial expansion, the number of standard deviaitons from which to start clipping outliers, other values in the preprocessing steps.

For each hyperparameter combination, the models were trained using K-flod cross validation. The average loss, accuracy, and F1 score over all lthe folds were used to select the best hyperparameters.

Once the best combination of hyperparameters was determined, the model was trained over the whole training set with these parameters. When modifying hyperparameters of the preprocessing steps, the preprocessing also had to be done from scartch.

At the end of the training, the model weights of the best model are determined. They will then be used for testing.

# Testing

The test data is loaded from the test.csv dataset and is preprocessed with the same steps as during the training.

From the trained model weights, labels are predicted on the test data. The labels and their correesponding id are then stored in the submission.csv file in the /dataset folder. That file can then be used for submission on the competition on AIcrowd and is the output of the program.

# Authors
Maxime Bourassa
Nitish Ravishankar
Agastya
