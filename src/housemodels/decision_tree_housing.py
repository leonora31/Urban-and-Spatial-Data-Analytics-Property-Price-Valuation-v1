import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


class DecisionTreeModel:
    def __init__(self, data, feat_cols):
        self.data = data
        self.feat_cols = feat_cols
        self.model = DecisionTreeRegressor(random_state=0)
        self.grid_search = None
        self.best_params = None

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
        parameters = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10]
        }

        best_mae = float('inf')  # initialize with a high value
        best_params = None

        for max_depth in parameters['max_depth']:
            for min_samples_split in parameters['min_samples_split']:
                # Initialize and train model with current parameters
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=0)
                model.fit(self.X_train, self.y_train)

                # Predict on validation set and compute MSE
                predictions = model.predict(self.X_val)
                mae = mean_absolute_error(predictions, self.y_val)

                # If current MAE is lower than best MAE, update best parameters
                if mae < best_mae:
                    best_mae = mae
                    best_params = {'max_depth': max_depth,
                                   'min_samples_split': min_samples_split}

        # Set the best parameters for the model
        self.best_params = best_params
        self.model = DecisionTreeRegressor(**self.best_params, random_state=0)

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        self.predict("test")

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
        return "DT"
