from dpipe.train.batch_iter import make_batch_iter_from_finite
from dpipe.tf.model import TFModel, TFFrozenModel
from dpipe.tf.utils import get_tf_optimizer, softmax, softmax_cross_entropy
from functools import partial
from dpipe.train.train import train_base
from dpipe.train.lr_policy import Constant
from dpipe.train.validator import validate
from sklearn.metrics import accuracy_score

from model_core.lenet import LeNet
from dataset.mnist import MNIST
from batch_iter.simple import simple_iterator

import numpy as np

# dataset
mnist = MNIST('../../MNIST_data/')

# batch iter
batch_iter = make_batch_iter_from_finite(
    get_batch_iter=partial(
        simple_iterator,
        ids=mnist.ids[:16*3000],
        load_x=mnist.load_image,
        load_y=mnist.load_label,
        batch_size=16,
        shuffle=False
    )
)

# model core
lenet = LeNet(28, 1, 10)

# optimizer
optimizer = partial(
    get_tf_optimizer,
    tf_optimizer_name='AdamOptimizer',
    beta1=0.899
)

# model tf
model = TFModel(lenet, softmax, softmax_cross_entropy, optimizer)

# accuracy
def accuracy_metric(y_true, y_one_hot):
    y_pred = np.array(y_one_hot).squeeze().argmax(axis=1)
    return accuracy_score(y_true, y_pred)

# val metrics
val_metrics = {}
val_metrics['accuracy'] = accuracy_metric

# validatate function
def validate_fn(x, y):
    try:
        l = len(y)
    except TypeError:
        x = x[np.newaxis, ...]
        y = np.array([y])
    return model.do_val_step(x, y)

validator = partial(
    validate,
    load_x=mnist.load_image,
    load_y=mnist.load_label,
    ids=mnist.ids[16 * 3000:],
    metrics=val_metrics,
    validate_fn=validate_fn
)

def main():
    # run train
    n_epochs = 5
    lr_init = 1e-3
    train_base(model, batch_iter, n_epochs, lr_policy=Constant(lr_init), log_path='logs/', validator=validator)

if __name__ == '__main__':
    main()