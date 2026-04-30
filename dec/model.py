import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import linear_sum_assignment


class AssignableDense(tf.Module):
    def __init__(self, input_size, units, activation=tf.nn.relu, name=None):
        super().__init__(name=name)
        self.activation = activation
        self.w = tf.Variable(tf.random.normal(shape=(input_size, units), stddev=0.01), name='w')
        self.b = tf.Variable(tf.zeros(shape=(units,)), name='b')

    def assign_weights(self, from_dense):
        self.w.assign(from_dense.w)
        self.b.assign(from_dense.b)

    def apply(self, x):
        out = tf.matmul(x, self.w) + self.b
        if self.activation is not None:
            out = self.activation(out)
        return out


class StackedAutoEncoder(tf.Module):
    def __init__(self, encoder_dims, input_dim, name=None):
        super().__init__(name=name)
        self.layerwise_autoencoders = []

        layer_dims = [input_dim] + encoder_dims
        for i in range(1, len(layer_dims)):
            if i == 1:
                sub_ae = AutoEncoder([layer_dims[i]], layer_dims[i - 1], decode_activation=False)
            elif i == len(layer_dims) - 1:
                sub_ae = AutoEncoder([layer_dims[i]], layer_dims[i - 1], encode_activation=False)
            else:
                sub_ae = AutoEncoder([layer_dims[i]], layer_dims[i - 1])
            self.layerwise_autoencoders.append(sub_ae)


class AutoEncoder(tf.Module):
    def __init__(self, encoder_dims, input_dim, encode_activation=True, decode_activation=True, name=None):
        super().__init__(name=name)
        self.encoder_dims = encoder_dims
        self.input_dim = input_dim
        self.dense_layers = []
        self._build_layers(input_dim, encoder_dims, encode_activation, decode_activation)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.1,
            decay_steps=20000,
            decay_rate=0.1,
            staircase=True,
        )
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    def _build_layers(self, input_dim, encoder_dims, encode_activation, decode_activation):
        layer_dims = [input_dim] + encoder_dims
        for i in range(len(encoder_dims)):
            in_size = layer_dims[i]
            out_size = layer_dims[i + 1]
            act = None if (not encode_activation and i == len(encoder_dims) - 1) else tf.nn.relu
            self.dense_layers.append(AssignableDense(in_size, out_size, activation=act))

        decoder_dims = list(reversed(encoder_dims[:-1])) + [input_dim]
        in_size = encoder_dims[-1]
        for i, out_size in enumerate(decoder_dims):
            act = None if (not decode_activation and i == len(decoder_dims) - 1) else tf.nn.relu
            self.dense_layers.append(AssignableDense(in_size, out_size, activation=act))
            in_size = out_size

    def encode(self, x, keep_prob=1.0):
        layer = x
        for i in range(len(self.encoder_dims)):
            layer = tf.nn.dropout(layer, rate=1.0 - keep_prob)
            layer = self.dense_layers[i].apply(layer)
        return layer

    def decode(self, z, keep_prob=1.0):
        layer = z
        n_enc = len(self.encoder_dims)
        for i in range(n_enc, len(self.dense_layers)):
            layer = tf.nn.dropout(layer, rate=1.0 - keep_prob)
            layer = self.dense_layers[i].apply(layer)
        return layer

    def train_step(self, x, keep_prob=0.8):
        with tf.GradientTape() as tape:
            z = self.encode(x, keep_prob)
            decoded = self.decode(z, keep_prob)
            loss = tf.reduce_mean(tf.square(x - decoded))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


class DEC(tf.Module):
    def __init__(self, params, name=None):
        super().__init__(name=name)
        self.n_cluster = params["n_clusters"]
        self.kmeans = KMeans(n_clusters=params["n_clusters"], n_init=20)
        self.ae = AutoEncoder(
            params["encoder_dims"], params["input_dim"],
            encode_activation=False, decode_activation=False,
        )
        self.alpha = params["alpha"]
        self.mu = tf.Variable(
            tf.zeros(shape=(params["n_clusters"], params["encoder_dims"][-1])),
            name="mu",
        )
        self.optimizer = tf.keras.optimizers.Adam(0.001)

    def get_assign_cluster_centers_op(self, features):
        print("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        print("Kmeans train end.")
        self.mu.assign(kmeans.cluster_centers_)

    def soft_assignment(self, x):
        z = self.ae.encode(x, keep_prob=1.0)
        return self._soft_assignment(z, self.mu)

    def _soft_assignment(self, embeddings, cluster_centers):
        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster)),
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(tf.shape(a)[0], 1)),
                transpose_b=True,
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target * tf.math.log(target / pred), axis=1))

    def train_step(self, x, p, keep_prob=0.8):
        with tf.GradientTape() as tape:
            z = self.ae.encode(x, keep_prob)
            q = self._soft_assignment(z, self.mu)
            loss = self._kl_divergence(p, q)
        grads = tape.gradient(loss, self.trainable_variables)
        grad_var_pairs = [(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(grad_var_pairs)
        pred = tf.argmax(q, axis=1)
        return loss, pred

    def cluster_acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
