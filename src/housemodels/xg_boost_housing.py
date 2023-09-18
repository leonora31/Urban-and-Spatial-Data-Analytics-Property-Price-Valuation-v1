import time
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error


class XGBoostModel:

    def __init__(self, data, feat_cols):
        self.data = data
        self.feat_cols = feat_cols
        self.model = xgb.XGBRegressor()
        self.best_params = None
        # Initialize with a high value
        self.best_mae = float('inf')
        self.best_model = None

        # Split and prepare the data immediately upon initialization
        self.prepare_data()

        # Initialize placeholders for original and predicted values
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

        # Encode categorical variables
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.X_train = self.encoder.fit_transform(self.X_train)
        self.X_val = self.encoder.transform(self.X_val)
        self.X_test = self.encoder.transform(self.X_test)

        # Standardize features
        self.scaler = StandardScaler(with_mean=False)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

    def tune(self):
        learning_rates = [0.01, 0.1, 0.2]
        n_estimators = [100, 200]
        max_depths = [3, 5]

        best_params = None

        # Total number of combinations
        total_combinations = (len(learning_rates) *
                              len(n_estimators) * len(max_depths))
        current_combination = 0

        for lr in learning_rates:
            for n_est in n_estimators:
                for depth in max_depths:
                    current_combination += 1
                    print(f"\nTraining combination {current_combination}"
                          f"/{total_combinations}...")
                    print(f"Parameters: learning_rate={lr}, "
                          f"n_estimators={n_est}, max_depth={depth}")
                    start_time = time.time()

                    model = xgb.XGBRegressor(learning_rate=lr,
                                             n_estimators=n_est,
                                             max_depth=depth)

                    model.fit(self.X_train, self.y_train)
                    preds = model.predict(self.X_val)
                    mae = mean_absolute_error(self.y_val, preds)

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    print("Finished training combination "
                          f"{current_combination}/{total_combinations} in "
                          f"{elapsed_time:.2f} seconds.")
                    print(f"MAE on validation set: {mae}\n")

                    # Update the best model and parameters if current model
                    # is better
                    if mae < self.best_mae:
                        self.best_mae = mae
                        self.best_model = model
                        best_params = {'learning_rate': lr, 'n_estimators':
                                       n_est, 'max_depth': depth}

        print(f"Best parameters found: {best_params}")
        print(f"Best MAE on validation set: {self.best_mae}")

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
        return "XGBoost"
