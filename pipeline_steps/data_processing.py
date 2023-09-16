import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


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
    path = f'datasets/processed_data/{ticker_symbol}'
    if not os.path.exists(path):
        os.mkdir(path)

    # write out file
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    df.to_csv(f'{path}/{timestamp}.csv', index=False)

    if info:
        print(f'File written out: {timestamp}.csv')
        print(f'Path to file: {path}/')