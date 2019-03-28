import tensorflow as tf

class softmax_cross_entropy:
    def __init__(self,weights=None):
        self.weights = weights
    def __call__(self,onehot_labels,logits):
        return tf.losses.softmax_cross_entropy(onehot_labels,logits)