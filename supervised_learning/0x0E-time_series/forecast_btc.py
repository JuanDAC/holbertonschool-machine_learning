#!/usr/bin/env python3
"""File that contains the function forecast_btc"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


preprocessor = __import__('preprocess_data').preprocess_data


class WindowGenerator:
    """
    class property that returns the training dataset as output
    """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        """
        This function initializes the class WindowGenerator with the following parameters:
        - input_width: the number of input time steps
        - label_width: the number of label time steps
        - shift: the number of time steps between the input and the label
        - train_df: the training dataframe
        - val_df: the validation dataframe
        - test_df: the testing dataframe
        - label_columns: the label columns
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def split_window(self, features):
        """
        This function splits the window into inputs and labels
        and returns them as output
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
        This function makes a dataset from the data given as input
        and returns it as output
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        """
        This function returns the training dataset as output
        """
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """
        This function returns the validation dataset as output
        """
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """
        This function returns the testing dataset as output
        """
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """
        This function get and cache an example batch of (inputs, labels) for plotting
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        """
        This function plots the model's predictions
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def plot_train_history(self, history, title):
        """
        This function plots the training history
        """
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(loss))

        plt.figure(figsize=(12, 8))
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title(title)
        plt.legend()


class Baseline(tf.keras.Model):
    """
    The class that represent general architecture of the model is constituted of an input layer
    of dimension (time steps × number of features = 90 × 24), the inner structure and an
    output layer of dimension (time steps × number of classes = 90 × 3). Indeed, we apply
    at each future time step a softmax layer with three neurons, one for each class of the
    output variables. We will change the inner structure of the model to test different
    architectures of neural networks.
    """

    def __init__(self, label_index=None):
        """
        This function initializes the class
        """
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        """
        This function defines the forward pass of the model
        """
        if self.label_index is None:
            result = inputs
        else:
            result = inputs[:, :, self.label_index]
        return tf.expand_dims(result, axis=-1)

    def compile(self, window, patience):
        """
          This function compile and fit the model
        """
        super().compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min')

        return self.fit(window.train, epochs=100,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    def build_model(self):
        """
        This function builds the model
        """
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.Dense(units=1)
        ])
        lstm_model.summary()
        return lstm_model


if __name__ == '__main__':
    pd.__version__
    train_df, val_df, test_df = preprocessor
    window = WindowGenerator(
        input_width=24, label_width=1, shift=1, label_columns=['Close'])
    column_indices = window.column_indices

    val_performance = {}
    performance = {}
    baseline = Baseline(label_index=column_indices['Close'])
    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])

    val_performance['Baseline'] = baseline.evaluate(window.val)
    performance['Baseline'] = baseline.evaluate(window.test, verbose=0)
    print(val_performance)
    print(performance)

    window.plot(baseline)
    lstm_model = baseline.build_model
    MAX_EPOCHS = 50
    history = baseline.compile_and_fit(lstm_model, window)

    # Evaluate LSTM with wide_window
    val_performance['LSTM'] = lstm_model.evaluate(window.val)
    performance['LSTM'] = lstm_model.evaluate(window.test, verbose=0)

    window.plot(lstm_model)
