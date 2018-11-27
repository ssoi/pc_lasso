import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin

class TFLasso(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.1, pc_penalty=False, batch_size=100, epochs=100, learning_rate=0.01):
        self.alpha = alpha
        self.pc_penalty = pc_penalty
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self._tf_model = None

    def fit(self, X, y):
        if self._tf_model is None:
            self._build_tf_model(X, y)

        
        optimizer = tf.GradientDescentOptimizer(learning_rate=self.learning_rate).\
            minimize(self._tf_model)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            num_batches = int(X.shape[0]/self.batch_size)
            for epoch in range(self.epochs):
                for _ in range(num_batches):
                    X_batch, y_batch = _get_batch(X, y, self.batch_size)
                    sess.run([optimizer, self._tf_model], feed_dict={X: X_batch, y: y_batch})

    def predict(self):
        pass

    def _build_tf_model(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of observations must match between X and y')

        num_obs, num_features = X.shape

        X_tensor = tf.placeholder(tf.float32, [self.batch_size, num_features], name='X')
        y_tensor = tf.placeholder(tf.float32, [self.batch_size, 1], name='y')
        
        beta = tf.Variable(tf.random_normal(shape=[num_features, 1], stddev=0.1), name='coefs')        
        beta_0 = tf.Variable(tf.random_normal(shape=[1, 1], stddev=0.1), name='intercept')

        logits = tf.matmul(X_tensor, beta) + beta_0
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy) + alpha * tf.norm(beta)

        self._tf_model = loss

    @staticmethod
    def _get_batch(X, y, batch_size):
        num_obs = X.shape[0]
        idx = np.random.choice(num_obs, batch_size)

        return X[idx, :], y[idx]
