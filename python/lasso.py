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
        self.coef_ = None

        self._X_tensor = None
        self._y_tensor = None
        self._beta = None
        self._beta_0 = None
        self._tf_model = None

    def fit(self, X, y):
        if self._tf_model is None:
            self._build_tf_model(self.batch_size, X.shape[1])

        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).\
            minimize(self._tf_model)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            num_batches = int(X.shape[0]/self.batch_size)
            for epoch in range(self.epochs):
                for _ in range(num_batches):
                    X_batch, y_batch = self._get_batch(X, y)
                    sess.run(
                        [optimizer, self._tf_model],
                        feed_dict={self._X_tensor: X_batch, self._y_tensor: y_batch}
                    )

            self.coef_ = (sess.run(self._beta), sess.run(self._beta_0))

    def predict(self):
        pass

    def _build_tf_model(self, num_obs, num_features):
        self._X_tensor = tf.placeholder(tf.float32, [num_obs, num_features], name='X')
        self._y_tensor = tf.placeholder(tf.float32, [num_obs, 1], name='y')
        
        self._beta = tf.Variable(tf.random_normal(shape=[num_features, 1], stddev=0.1), name='coefs')        
        self._beta_0 = tf.Variable(tf.random_normal(shape=[1, 1], stddev=0.1), name='intercept')

        logits = tf.matmul(self._X_tensor, self._beta) + self._beta_0
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y_tensor, logits=logits)
        loss = tf.reduce_mean(cross_entropy) + self.alpha * tf.abs(self._beta)

        self._tf_model = loss

    def _get_batch(self, X, y):
        num_obs = X.shape[0]
        idx = np.random.choice(num_obs, self.batch_size)

        y_batch = y[idx].reshape(self.batch_size, 1)
        X_batch = X[idx, :]

        return X_batch, y_batch
