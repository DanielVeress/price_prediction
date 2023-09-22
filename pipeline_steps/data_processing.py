import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import PROCESSED_DATA_DIR

def load_data_for_ticker(ticker_symbol, file, info=0):
    ''' 
    Loads in data for a specified company ticker
    '''
    df = pd.read_csv(file)

    # setting date as the index
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)

    # sort by date, just in case
    df.sort_index(ascending=True, inplace=True)

    # add ticker as column (if it isn't there)
    if 'ticker' not in df.columns:
        df['ticker'] = ticker_symbol

    if info:
        # information about our data
        print(f'Rows: {df.shape[0]}')
        print(f'Columns: {df.shape[1]}')
        start_date_str = df.index[0].strftime(f'%Y/%m/%d')
        end_date_str = df.index[df.shape[0]-1].strftime(f'%Y/%m/%d')
        print(f'Date range: {start_date_str} - {end_date_str}')
        
        # plot for data
        plt.figure(figsize = (18,9))
        plt.plot(df.index, df['Close'])
        plt.show()

    return df


def write_processed_data(df, info=0):
    '''
    Writes out the processed dataframe to a csv
    '''
    
    # setting up folder for ticker, if there isn't any
    ticker_symbol = df['ticker'][0]
    path = f'{PROCESSED_DATA_DIR}/{ticker_symbol}'
    if not os.path.exists(path):
        os.mkdir(path)

    # write out file
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    df.to_csv(f'{path}/{timestamp}.csv', index=False)

    if info:
        print(f'File written out: {timestamp}.csv')
        print(f'Path to file: {path}/')


def __calc_target(batch, calc_type='simple_diff'):
    '''Calculates one target value for a batch of future values.'''

    batch = batch.to_numpy()
    current_value = batch[0]
    future_values = batch[1:]
    
    value = 0
    match calc_type:
        case 'simple_diff':
            value = future_values[-1]
        case 'mean_diff':
            value = sum(future_values)/len(future_values)
        case _:
            print(f"WARNING: {calc_type} deosn't exist!s")
    
    return value - current_value


def create_target_column(column, difference_range, calc_type='simple_diff'):
    '''Creates a target column from future prices.'''

    reversed_col = column[::-1]
    
    # go through the column with the window (+1 for the current price)
    result = reversed_col.rolling(window=difference_range+1).apply(lambda x: __calc_target(x[::-1], calc_type))
    return result[::-1]


def train_val_test_split(X, y, train_prec, val_prec, test_prec, info=1):
    '''
    Splits data into train, validation and test sets.
    '''

    if train_prec+val_prec+test_prec != 1:
        print("WARNING: The given precentages didn't add up to 1! Please check the given parameters!")

    train_size = int(X.shape[0] * train_prec)
    val_size = int(X.shape[0] * val_prec)

    # splitting X
    X_train = X[0:train_size]
    X_val = X[train_size:(train_size+val_size)]
    X_test = X[(train_size+val_size):]
    
    # splitting Y (target values)
    y_train = y[0:train_size]
    y_val = y[train_size:(train_size+val_size)]
    y_test = y[(train_size+val_size):]

    # make them into numpy arrays
    X_train, X_val, X_test = X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy()
    y_train, y_val, y_test = y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()

    if info:
        print('Dataset ranges: ')
        print(f'\tTrain: {0}-{train_size-1}')
        print(f'\tVal: {train_size}-{train_size+val_size-1}')
        print(f'\tTest: {train_size+val_size}-{X.shape[0]-1}')

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_sequence(X, y, sequence_length, info=1):
    '''
    Creates sequences that can be fed to LSTM like models.
    '''    
   
    # removing 1 window's size, so we don't go over the data's size
    data_size = X.shape[0] - sequence_length
    
    X_seq = []
    y_seq = []
    for i in range(0, data_size):
        # adding 'sequence lenght' amount of data points
        X_seq.append(X[i:i+sequence_length])
        
        # adding target value from last data point in the sequence
        y_seq.append(y[i+sequence_length-1])  

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if info:
        print(f'Original shape: \n\tX:{X.shape} and y:{y.shape}')
        print(f'Sequence shape: \n\tX:{X_seq.shape} and y:{y_seq.shape}')
        
    return X_seq, y_seq


def normalize_data(X_train, X_val, X_test, scaler, grouping_precentage=0.1, info=1, verbose=0):
    '''
    Normalizes data inplace with a given scaler.
    
    Note: uses windows for normalization, to not use windows set the grouping_precentage to 1
    '''
    
    # getting 1 normalization window's size
    all_data_size = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    normalization_window_size = math.ceil(all_data_size * grouping_precentage)
    if info:
        print(f'Full data size: {all_data_size}')
        print(f"Normalization window's size: {normalization_window_size}")

    if verbose: print('Data normalization started...')

    # normalizing training data
    for i in range(0, X_train.shape[0], normalization_window_size):
        window_end = i + normalization_window_size
        X_train[i:window_end] = scaler.fit_transform(X_train[i:window_end])
        
        if verbose: 
            print(f'\tNormalized training data from {i} to {window_end}')
            if window_end > X_train.shape[0]: print(f'\t Training data end was at {X_train.shape[0]}')
        
    # normalizing validation data
    for i in range(0, X_val.shape[0], normalization_window_size):
        window_end = i + normalization_window_size
        X_val[i:window_end] = scaler.transform(X_val[i:window_end])

        if verbose: 
            print(f'\tNormalized validation data from {i} to {window_end}')
            if window_end > X_val.shape[0]: print(f'\t Training data end was at {X_val.shape[0]}')
        
    # normalizing test data (without fitting the scaler to it!)
    for i in range(0, X_test.shape[0], normalization_window_size):
        window_end = i + normalization_window_size
        X_test[i:window_end] = scaler.transform(X_test[i:window_end])

        if verbose: 
            print(f'\tNormalized testing data from {i} to {window_end}')
            if window_end > X_test.shape[0]: print(f'\t Training data end was at {X_test.shape[0]}')