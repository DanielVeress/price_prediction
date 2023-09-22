import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import mlflow

from pipeline_steps.model_prediction import calc_metrics
from utils.constants import SEED

tf.random.set_seed(SEED)

class OptimizerPipeline():
    epochs = 50
    batch_size = 128
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    best_metric = None
    best_model = None

    def __init__(self, trial_number, model_type, optimized_metric, metric_direction, 
                 sequence_length, feature_num, X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq):
        self.model_type = model_type
        self.optim_trial_num = trial_number
        self.chosen_metric_name = optimized_metric
        self.metric_direction = metric_direction

        self.sequence_length = sequence_length
        self.feature_num = feature_num
        self.X_train_seq = X_train_seq
        self.y_train_seq = y_train_seq
        self.X_val_seq = X_val_seq
        self.y_val_seq = y_val_seq
        self.X_test_seq = X_test_seq
        self.y_test_seq = y_test_seq


    def save_model(self, metric, model):
        if self.best_metric == None:
            self.best_metric = metric
            self.best_model = model
        elif self.metric_direction == 'minimize' and self.best_metric > metric:
            self.best_metric = metric
            self.best_model = model
        elif self.metric_direction == 'maximize' and self.best_metric < metric:
            self.best_metric = metric
            self.best_model = model


    def choose_optimizer(self, trial):
        optimizers = ['adam']
        optimizer_name = trial.suggest_categorical('optimizers', optimizers)

        return optimizer_name


    def choose_loss(self, trial):
        loss_functions = ['mean_squared_error']
        loss_name = trial.suggest_categorical('loss_function', loss_functions)

        return loss_name


    def add_layer(self, trial, model, layer_idx, units, activation='relu', return_sequences=False):
        layer_type = self.model_type
        if (self.model_type == 'HYBRID'):
            layer_type = trial.suggest_categorical('type_l{}'.format(layer_idx),['LSTM', 'BLSTM', 'GRU'])

        match layer_type:
            case 'LSTM':
                model.add(LSTM(units, activation, input_shape=(self.sequence_length, self.feature_num), return_sequences=return_sequences))
            case 'BLSTM':
                model.add(Bidirectional(LSTM(units, activation, input_shape=(self.sequence_length, self.feature_num), return_sequences=return_sequences)))
            case 'GRU':
                model.add(GRU(units, activation, input_shape=(self.sequence_length, self.feature_num), return_sequences=return_sequences))


    def create_model(self, trial):
        n_layers = trial.suggest_int('n_layers', 1, 5)
        
        # adding units for every layer
        units = []
        for i in range(n_layers):
            n_units = trial.suggest_int('n_units_l{}'.format(i), 8, 128, step=8)
            units.append(n_units)

        # creating model and its layers
        model = Sequential()
        for i in range(n_layers):
            if i == n_layers-1:
                self.add_layer(trial, model, i, units[i])
            else:   
                self.add_layer(trial, model, i, units[i], return_sequences=True)
            model.add(Dropout(0.2))
        model.add(Dense(1))
        
        # compiling model
        optimizer = self.choose_optimizer(trial)
        loss = self.choose_loss(trial)
        model.compile(optimizer=optimizer, loss=loss)

        return model


    def objective(self, trial):
        model = self.create_model(trial)
        
        # train model
        history = model.fit(
            self.X_train_seq, self.y_train_seq, 
            epochs=self.epochs, batch_size=self.batch_size,
            validation_data=(self.X_val_seq, self.y_val_seq),
            callbacks=self.callbacks)
        
        # predict
        y_pred = model.predict(self.X_test_seq).reshape(-1)
        y_true = self.y_test_seq
        metrics = calc_metrics(y_true, y_pred)

        # set metric that we are optimizing by
        if self.chosen_metric_name == 'val_loss':
            observed_metric = min(history.history['val_loss'])
        else:
            observed_metric = metrics[self.chosen_metric_name]
        self.save_model(observed_metric, model)

        return observed_metric


    def run(self):
        study = optuna.create_study(direction=self.metric_direction)
        study.optimize(self.objective, n_trials=self.optim_trial_num, timeout=600)

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        self.__mlflow_logging(trial)

        return trial
    

    def __mlflow_logging(self, trial):
        mlflow.log_param('trial_numbers', self.optim_trial_num)
        mlflow.log_param('chosen_metric_name', self.chosen_metric_name)
        mlflow.log_param('model_type', self.model_type)
        mlflow.log_param('epochs', self.epochs)
        mlflow.log_param('batch_size', self.batch_size)

        for key, value in trial.params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric(self.chosen_metric_name, self.best_metric)

        self.best_model.save('models/model.h5')
        mlflow.log_artifact(local_path='models/model.h5', artifact_path='model')