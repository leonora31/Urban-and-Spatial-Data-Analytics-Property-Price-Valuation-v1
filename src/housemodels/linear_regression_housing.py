import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


class LinearRegressionModel:
    def __init__(self, data, feat_cols):
        self.data = data
        self.feat_cols = feat_cols
        self.model = LinearRegression()
        self.best_params = None
        self.best_mae = float('inf')  # Initialize with a high value
        self.best_model = None

        # Split and prepare the data immediately upon initialization
        self.prepare_data()

        # Initialize placeholders for original and predicted values
        # Storing the original y_test
        self.original_y_test = np.exp(self.y_test)
        self.predictions_test = None

    def prepare_data(self):
        # Filter based on years to create train, validation, and test splits
        self.X_train = self.data[self.data['Year'] <= 2017][self.feat_cols]
        self.X_val = self.data[self.data['Year'] == 2018][self.feat_cols]
        self.X_test = self.data[self.data['Year'] == 2019][self.feat_cols]

        self.y_train = self.data[self.data['Year'] <= 2017]['Price']
        self.y_val = self.data[self.data['Year'] == 2018]['Price']
        self.y_test = self.data[self.data['Year'] == 2019]['Price']

        # Normalize the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

    def tune(self):
        # Parameter values to try
        positive_options = [True, False]

        best_params = {}

        for positive in positive_options:
            # Create and train a model with the current parameter
            params = {'positive': positive}
            model = LinearRegression(**params)
            model.fit(self.X_train, self.y_train)

            # Predict on the validation set and compute MAE
            y_pred_val = model.predict(self.X_val)
            mae = mean_absolute_error(self.y_val, y_pred_val)

            # Update the best model and parameters if current model is better
            if mae < self.best_mae:
                self.best_mae = mae
                self.best_model = model
                best_params = params  # Update best parameters

        print(f"Best MAE on validation set: {self.best_mae}")
        print(f"Best parameters: {best_params}")
        self.best_params = best_params
        self.model = self.best_model  # Use the best model found during tuning

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        self.predict(dataset='test')

    def predict(self, dataset='val'):
        if dataset == 'val':
            predictions = self.model.predict(self.X_val)
            # Convert from log scale to original and store
            self.predictions_val = np.exp(predictions)
        else:
            predictions = self.model.predict(self.X_test)
            # Convert from log scale to original and store
            self.predictions_test = np.exp(predictions)
        return predictions

    def __str__(self):
        return "LR"
