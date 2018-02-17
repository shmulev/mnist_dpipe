from sklearn.metrics import accuracy_score
import numpy as np

def accuracy_metric(y_true, y_one_hot):
    y_pred = np.array(y_one_hot).squeeze().argmax(axis=1)
    return accuracy_score(y_true, y_pred)


val_metrics = {}
val_metrics['accuracy'] = accuracy_metric


def validate_fn(x, y, *, model):
    try:
        l = len(y)
    except TypeError:
        x = x[np.newaxis, ...]
        y = np.array([y])
    return model.do_val_step(x, y)
