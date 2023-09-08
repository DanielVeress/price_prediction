import numpy as np


def format_num(num, infix="'", postfix='$'):
    reversed_num = str(num)[::-1]
    formatted_num = ''
    grouping = 3 # digits
    
    group_idx = 0
    for letter in reversed_num:
        if letter.isdigit():
            if group_idx == grouping:
                formatted_num = infix + formatted_num
                group_idx = 0
            formatted_num = letter + formatted_num
            group_idx += 1
        else:
            formatted_num = letter + formatted_num
    
    return formatted_num + postfix


def train_val_test_split(X, y, train_prec, val_prec, test_prec, verbose=0):
    if train_prec+val_prec+test_prec != 1:
        print("WARNING: The given precentages didn't add up to 1! Please check the given parameters!")

    train_size = int(X.shape[0] * train_prec)
    val_size = int(X.shape[0] * val_prec)

    X_train = X[0:train_size]
    X_val = X[train_size:(train_size+val_size)]
    X_test = X[(train_size+val_size):]
    
    y_train = y[0:train_size]
    y_val = y[train_size:(train_size+val_size)]
    y_test = y[(train_size+val_size):]

    if verbose:
        print('Split ranges: ')
        print(f'\tTrain: {0}-{train_size-1}')
        print(f'\tVal: {train_size}-{train_size+val_size-1}')
        print(f'\tTest: {train_size+val_size}-{X.shape[0]-1}')

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_sequence(X, y, sequence_legnth, verbose=0):
    '''
    Creates sequences that can be fed to LSTM like models.
    '''    
    # removing last window's size
    data_size = X.shape[0] - sequence_legnth
    
    X_seq = []
    y_seq = []
    for i in range(0, data_size):
        # adding 'sequence lenght' amount of data points
        X_seq.append(X[i:i+sequence_legnth])
        
        # adding target value from last data point in the sequence
        y_seq.append(y[i+sequence_legnth-1])  

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if verbose:
        print(f'Original shape: \n\tX:{X.shape} and y:{y.shape}')
        print(f'Sequence shape: \n\tX:{X_seq.shape} and y:{y_seq.shape}')
        
    return X_seq, y_seq