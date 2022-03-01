from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import initializers
import numpy as np
from keras import backend as K

from utils import create_model_checkpoint, plot_ensemble_history


class AnchoredNet:
    """ Define and build a simple feed forward neural net"""
    def __init__(self, descriptor_size: int = 1024, hidden_size: int = 512, dropout: float = 0.5, lr: float = 0.0001,
                 regularization: float = 0.001, batch_size=64, patience_stopping: int = 20, monitor: str = 'val_loss',
                 patience_lr: int = 10, lr_factor: float = 0.5, min_lr: float = 1.0e-05, random_seed: int = 42,
                 prior_var: float = 0.001):

        self.model = None
        self.descriptor_size = descriptor_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.regular_l2 = regularization
        self.lr = lr
        self.batch_size = batch_size
        self.seed = random_seed

        # Callbacks
        self.early_stopping = EarlyStopping(monitor=monitor, mode='min', verbose=1, patience=patience_stopping)
        self.checkpointer = create_model_checkpoint()
        self.lr_reduction = ReduceLROnPlateau(monitor=monitor, patience=patience_lr, factor=lr_factor, min_lr=min_lr)

        # Init weights and biases
        self.W1_anchor = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.descriptor_size, self.hidden_size])
        self.W1_init = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.descriptor_size, self.hidden_size])
        self.b1_anchor = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size])
        self.b1_init = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size])

        self.W2_anchor = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size, self.hidden_size])
        self.W2_init = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size, self.hidden_size])
        self.b2_anchor = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size])
        self.b2_init = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size])

        self.W3_anchor = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size, self.hidden_size])
        self.W3_init = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size, self.hidden_size])
        self.b3_anchor = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size])
        self.b3_init = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size])

        self.W_last_anchor = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size, 1])
        self.W_last_init = np.random.normal(loc=0, scale=np.sqrt(prior_var), size=[self.hidden_size, 1])

        self.build_model()

    # create custom anchored regularization function
    def anchored_reg_W1(self, w):
        return K.sum(K.square(w - self.W1_anchor)) * self.regular_l2

    def anchored_reg_b1(self, w):
        return K.sum(K.square(w - self.b1_anchor)) * self.regular_l2

    def anchored_reg_W2(self, w):
        return K.sum(K.square(w - self.W2_anchor)) * self.regular_l2

    def anchored_reg_b2(self, w):
        return K.sum(K.square(w - self.b2_anchor)) * self.regular_l2

    def anchored_reg_W3(self, w):
        return K.sum(K.square(w - self.W2_anchor)) * self.regular_l2

    def anchored_reg_b3(self, w):
        return K.sum(K.square(w - self.b2_anchor)) * self.regular_l2

    def anchored_reg_W_last(self, w):
        return K.sum(K.square(w - self.W_last_anchor)) * self.regular_l2

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_size, activation="relu", input_shape=(self.descriptor_size,),
                       kernel_initializer=initializers.Constant(value=self.W1_init),
                       bias_initializer=initializers.Constant(value=self.b1_init),
                       kernel_regularizer=self.anchored_reg_W1,
                       bias_regularizer=self.anchored_reg_b1))
        self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.hidden_size, activation="relu", input_dim=self.descriptor_size,
                             kernel_initializer=initializers.Constant(value=self.W2_init),
                             bias_initializer=initializers.Constant(value=self.b2_init),
                             kernel_regularizer=self.anchored_reg_W2,
                             bias_regularizer=self.anchored_reg_b2))
        self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.hidden_size, activation="relu", input_dim=self.descriptor_size,
                             kernel_initializer=initializers.Constant(value=self.W3_init),
                             bias_initializer=initializers.Constant(value=self.b3_init),
                             kernel_regularizer=self.anchored_reg_W3,
                             bias_regularizer=self.anchored_reg_b3))
        self.model.add(Dropout(self.dropout))

        self.model.add(Dense(1, activation='linear', use_bias=False,
                             kernel_initializer=initializers.Constant(value=self.W_last_init),
                             kernel_regularizer=self.anchored_reg_W_last))

        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

    def train_model(self, x_train, y_train, x_val, y_val, n_workers: int = 4, verbose: bool = True, epochs: int = 100):

        self.history = self.model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), use_multiprocessing=True,
                                      callbacks=[self.checkpointer, self.lr_reduction, self.early_stopping],
                                      workers=n_workers, verbose=verbose, epochs=epochs)


def anchord_ensemble(x_train, y_train, x_val, y_val, x_test, n_ensembles: int = 10, regularization: float = 0.001,
                     epochs: int = 100):

    ensembles = []
    for i in range(n_ensembles):
        mod = AnchoredNet(regularization=regularization)
        mod.train_model(x_train, y_train, x_val, y_val, epochs=epochs)
        ensembles.append(mod)

    plot_ensemble_history(ensembles)

    y_preds = []
    for m in ensembles:
        y_preds.append(m.model.predict(x_test, verbose=0))

    y_preds = np.array(y_preds)
    y_preds = y_preds.reshape((len(y_preds[0]), len(y_preds)))
    y_preds_mean = np.mean(y_preds, axis=1)
    y_preds_std = np.std(y_preds, axis=1)

    return y_preds, y_preds_mean, y_preds_std












