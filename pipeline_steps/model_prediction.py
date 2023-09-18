import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

import plotly.graph_objects as go


def calc_metrics(y_true, y_pred):
    '''
    Calculates metrics for the predicted values.

    Returns a dictionary with the metrics.
    '''

    results = {}

    # MAE - Mean Absolute Error
    results['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
    
    results['MSE'] = metrics.mean_squared_error(y_true, y_pred)
    results['max-error'] = metrics.max_error(y_true, y_pred)

    # R^2 - coefficient of determination (should be close to 1)
    results['R2'] = metrics.r2_score(y_true, y_pred)

    return results


def plot_true_vs_predicted(X_plot, y_true, y_pred):
    fig = go.Figure()

    # adding true change
    fig.add_trace(
        go.Scatter(
            x=X_plot['Date'], 
            y=y_true,
            name='True change',
            line_color='blue',
        )
    )

    # adding predicted change
    fig.add_trace(
        go.Scatter(
            x=X_plot['Date'], 
            y=y_pred,
            name='Predicted change',
            line_color='orange'
        )
    )
    
    fig.update_yaxes(
        title_text='Predicted change', 
        tickprefix='$',
    )

    fig.update_layout(
        title='True change vs predicted change',
        title_x=0.5,
        height=600,
        width=1000,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        ),
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.2
    )

    fig.show()
    return fig


def plot_difference(X_plot, y_true, y_pred):
    fig = go.Figure()

    difference = y_pred - y_true
    fig.add_trace(
        go.Bar(
            x=X_plot['Date'],
            y=difference,
            marker=dict(color='red')
        )
    )

    fig.update_yaxes(
        title_text='Difference', 
        tickprefix='$',
    )

    fig.update_layout(
        title='Difference between real and predicted values',
        title_x=0.5,
        height=600,
        width=1000,
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.2
    )

    fig.show()
    return fig


def plot_prices(X_plot, y_true, y_pred):
    y_true_future_price = X_plot['Close'] + y_true
    y_pred_future_price = X_plot['Close'] + y_pred 

    fig = go.Figure()

    # adding true price
    fig.add_trace(
        go.Scatter(
            x=np.arange(y_true_future_price.shape[0]), 
            y=y_true_future_price,
            name='Real future price',
            line_color='blue',
        )
    )

    # adding predicted
    fig.add_trace(
        go.Scatter(
            x=np.arange(y_pred_future_price.shape[0]), 
            y=y_pred_future_price,
            name='Predicted future price',
            line_color='orange'
        )
    )

    fig.update_yaxes(
        title_text='Future price', 
        tickprefix='$'
    )

    fig.update_layout(
        title='Results',
        height=1200,
        title_x=0.5,
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.1
    )

    fig.show() 
    return fig