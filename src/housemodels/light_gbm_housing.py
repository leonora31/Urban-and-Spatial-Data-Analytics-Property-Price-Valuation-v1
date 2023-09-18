import numpy as np
import time
import lightgbm as lgb

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error


class LightGBMModel:

    def __init__(self, data, feat_cols):
        self.data = data
        self.feat_cols = feat_cols
        self.best_params = None
        self.best_mae = float('inf')  # Initialize with a high value
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

    def train(self):
        # Convert data to LightGBM Dataset format
        train_data = lgb.Dataset(self.X_train, label=self.y_train)

        # Define parameters (can be tuned further)
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
        }

        # Train LightGBM model
        self.model = lgb.train(params, train_data, num_boost_round=1000)
        self.predict(dataset='test')

    def tune(self):
        learning_rates = [0.01, 0.05, 0.1]
        num_leaves_list = [31, 40, 50]
        max_depths = [-1, 5, 10]

        # Initialize with a high value
        best_mae = float('inf')
        best_params = None

        # Convert data to LightGBM Dataset format
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        valid_data = lgb.Dataset(self.X_val, label=self.y_val,
                                 reference=train_data)

        # Total number of combinations
        num_lrs = len(learning_rates)
        num_ll = len(num_leaves_list)
        num_md = len(max_depths)
        total_combinations = num_lrs * num_ll * num_md
        current_combination = 0

        for lr in learning_rates:
            for num_leaves in num_leaves_list:
                for depth in max_depths:
                    current_combination += 1
                    print(f"Training combination {current_combination}"
                          f"/{total_combinations}...")
                    print(f"Parameters: learning_rate={lr},"
                          f" num_leaves={num_leaves}, max_depth={depth}")

                    params = {
                        'objective': 'regression',
                        'metric': 'mae',
                        'boosting_type': 'gbdt',
                        'learning_rate': lr,
                        'num_leaves': num_leaves,
                        'max_depth': depth
                    }

                    start_time = time.time()
                    model = lgb.train(params, train_data, num_boost_round=100,
                                      valid_sets=[valid_data])
                    preds = model.predict(self.X_val,
                                          num_iteration=model.best_iteration)
                    mae = mean_absolute_error(self.y_val, preds)

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    print(f"Finished train combination {current_combination}"
                          f"/{total_combinations} in "
                          f"{elapsed_time:.2f} seconds.")
                    print(f"MAE on validation set: {mae}\n")

                    # Update best parameters if current combination is better
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {'learning_rate': lr,
                                       'num_leaves': num_leaves,
                                       'max_depth': depth}
                        self.model = model

        print(f"Best parameters found: {best_params}")
        print(f"Best MAE on validation set: {best_mae}")

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
        return "LightGBM"
