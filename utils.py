

def create_model_checkpoint(save_path: str = '.'):
    """ Function to save the trained model during training """
    from tensorflow.keras.callbacks import ModelCheckpoint
    import os

    filepath = os.path.join(save_path + 'best_model.h5')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)

    return checkpointer


def plot_history(history, loss='mean_squared_error'):
    import matplotlib.pyplot as plt
    import numpy as np
    from math import ceil as ceiling

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(loss)
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, ceiling(max(history.history['loss']))])
    plt.show()


def plot_ensemble_history(ensembles, loss='mean_squared_error'):
    import matplotlib.pyplot as plt
    import numpy as np
    from math import ceil as ceiling

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(loss)

    for m in ensembles:

        plt.plot(m.history.epoch, np.array(m.history.history['loss']),
                 label='Train Loss', color='orange')
        plt.plot(m.history.epoch, np.array(m.history.history['val_loss']),
                 label='Val loss', color='blue')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.ylim([0, ceiling(max(m.history.history['loss']))])
    plt.show()


def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error
    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)
    Returns: (float) rmse
    """
    import numpy as np
    return np.sqrt(np.mean(np.square(np.array(true) - np.array(pred))))