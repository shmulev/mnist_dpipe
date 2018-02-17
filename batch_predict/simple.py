from dpipe.batch_predict.base import BatchPredict

class SimpleBatchPredict(BatchPredict):
    def predict(x, *, predict_fn):
        return predict_fn([x])