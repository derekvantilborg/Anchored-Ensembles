from Model import AnchoredNet, anchord_ensemble
from data_prep import prep_data
from utils import calc_rmse

# Get data
x_train, y_train, x_val, y_val, x_test, y_test = prep_data()

y_preds, y_preds_mean, y_preds_std = anchord_ensemble(x_train, y_train, x_val, y_val, x_test, n_ensembles=10)

calc_rmse(y_test, y_preds_mean)







