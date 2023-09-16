import math
import numpy as np


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
