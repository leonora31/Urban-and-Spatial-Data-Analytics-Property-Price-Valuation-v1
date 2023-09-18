import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers.legacy import Adam


class RNNModel:

    def __init__(self, data, features, target='Price', sequence_length=12):
        self.data = data
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

        self._prepare_data()

    def _prepare_data(self):
        # Sort data chronologically
        data = self.data.sort_values(by=['Year'])

        # Split data based on year
        train_data = data[data['Year'] < 2018]
        val_data = data[data['Year'] == 2018]
        test_data = data[data['Year'] == 2019]

        # Create sequences for each set
        self.X_train, self.y_train = self._create_sequences(train_data)
        self.X_val, self.y_val = self._create_sequences(val_data)
        self.X_test, self.y_test_original = self._create_sequences(
            test_data, return_original=True)

        # Scale data
        self._scale_data()

    def _scale_data(self):
        # Flatten, scale, and then reshape
        self.X_train = self.scaler.fit_transform(
            self.X_train.reshape(-1, len(self.features))
            ).reshape(self.X_train.shape)
        self.X_val = self.scaler.transform(
            self.X_val.reshape(-1, len(self.features))
            ).reshape(self.X_val.shape)
        self.X_test = self.scaler.transform(
            self.X_test.reshape(-1, len(self.features))
            ).reshape(self.X_test.shape)

    def train(self, epochs=10, verbose=1, **params):

        if params:
            best_params = {
                'learning_rate': None,
                'activation': None,
                'rnn_neurons': None,
                'batch_size': None,
                'dropout': None
            }
            best_params.update(params)

            # Extract values from best_params and use in your model
            learning_rate = best_params['learning_rate'] or 0.01
            activation = best_params['activation'] or 'relu'
            rnn_neurons = best_params['rnn_neurons'] or 50
            batch_size = best_params['batch_size'] or 32  # default batch size
            dropout = best_params['dropout'] or 0.0  # No dropout by default
        else:
            learning_rate = 0.01
            activation = 'relu'
            rnn_neurons = 50
            # default value.
            batch_size = 32
            dropout = 0.0

        model = Sequential()
        model.add(SimpleRNN(rnn_neurons, activation=activation,
                            dropout=dropout, input_shape=(self.sequence_length,
                                                          len(self.features))))
        model.add(Dense(1))
        self.load_initial_weights(model)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mae')

        model.fit(self.X_train, self.y_train,
                  validation_data=(self.X_val, self.y_val), epochs=epochs,
                  batch_size=batch_size, verbose=verbose, shuffle=False)
        y_pred_log = model.predict(self.X_test).squeeze()
        y_pred = np.exp(y_pred_log)  # convert from log to original scale
        self.predictions = y_pred
        self.model = model

    def tune(self, learning_rates=[0.001], activations=['relu'],
             rnn_neurons=[50], batch_sizes=[32, 64], dropouts=[0.0, 0.2]):
        best_mae = float('inf')
        best_params = None

        total_combinations = (len(learning_rates) *
                              len(activations) *
                              len(rnn_neurons) *
                              len(batch_sizes) *
                              len(dropouts))

        print(f"Total combinations: {total_combinations}")

        combination_count = 0

        for lr in learning_rates:
            for activation in activations:
                for neuron in rnn_neurons:
                    for batch_size in batch_sizes:
                        for dropout in dropouts:
                            combination_count += 1
                            print(f"\nTraining combination {combination_count}"
                                  f"/{total_combinations}...")
                            print(
                                f"Parameters: learning_rate={lr}, "
                                f"activation={activation}, "
                                f"rnn_neurons={neuron}, "
                                f"batch_size={batch_size}, "
                                f"dropout={dropout}"
                            )
                            start_time = time.time()

                            # Define the model with the current parameters
                            model = Sequential()
                            model.add(
                                SimpleRNN(neuron,
                                          activation=activation,
                                          input_shape=(self.sequence_length,
                                                       len(self.features)),
                                          dropout=dropout))
                            model.add(Dense(1))
                            self.load_initial_weights(model)
                            optimizer = Adam(learning_rate=lr)
                            model.compile(optimizer=optimizer, loss='mse')

                            # Train the model
                            model.fit(self.X_train, self.y_train,
                                      validation_data=(self.X_val, self.y_val),
                                      epochs=5, verbose=0, 
                                      batch_size=batch_size, shuffle=False)

                            # Predict on the validation set
                            y_pred_log = model.predict(self.X_val).squeeze()
                            y_pred = np.exp(y_pred_log)

                            mae = mean_absolute_error(np.exp(self.y_val),
                                                      y_pred)

                            # If the current model is better, update best_mae
                            # and best_params
                            if mae < best_mae:
                                best_mae = mae
                                best_params = {
                                    'learning_rate': lr,
                                    'activation': activation,
                                    'rnn_neurons': neuron,
                                    'batch_size': batch_size,
                                    'dropout': dropout
                                }
                                self.model = model

                            end_time = time.time()
                            time_taken = end_time - start_time

                            print("Finished training combination"
                                  f"{combination_count}/{total_combinations} "
                                  f"in {time_taken:.2f} seconds.")
                            print(f"MAE on validation set: {mae}")

        print(f"\nBest MAE on validation set: {best_mae}")
        print(f"Best parameters: {best_params}")

    # Convert the time-series data into a format suitable for training
    # Recurrent Neural Networks
    def _create_sequences(self, data, return_original=False):
        X = data[self.features].values
        y = data[self.target].values

        X_list, y_list, idx_list = [], [], []
        for i in range(len(data) - self.sequence_length):
            X_list.append(X[i:i+self.sequence_length])
            y_list.append(y[i+self.sequence_length])
            # Store the original index
            idx_list.append(data.index[i+self.sequence_length])

        X_seq = np.array(X_list)
        y_seq = np.array(y_list)

        if return_original:
            # Convert y_seq back to pandas Series
            y_seq = pd.Series(y_seq, index=data.index[self.sequence_length:])
            return X_seq, y_seq

        return X_seq, np.log(y_seq)  # apply log transformation

    def evaluate(self):
        y_pred_log = self.model.predict(self.X_test).squeeze()
        y_pred = np.exp(y_pred_log)  # convert from log to original scale

        # Calculate metrics on the original scale
        mae = mean_absolute_error(self.y_test_original, y_pred)
        r2 = r2_score(self.y_test_original, y_pred)

        print(f"MAE: {mae}")
        print(f"R^2 (Accuracy): {r2}")

        # Store original and prediction values
        self.predictions = y_pred

    def save_model(self, filename='rnn_model.h5'):
        self.model.save(filename)

    def __str__(self):
        return "RNN"

    def initialize_weights(self):
        """Initialize and save the model weights."""
        dummy_model = Sequential()
        dummy_model.add(SimpleRNN(50, activation='relu',
                                  input_shape=(self.sequence_length,
                                               len(self.features))))
        dummy_model.add(Dense(1))
        dummy_model.save_weights('initial_weights.h5')

    def load_initial_weights(self, model):
        """Load initial weights into the model."""
        # model.load_weights('initial_weights.h5')
        return model
