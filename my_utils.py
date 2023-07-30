import tensorflow as tf
from tensorflow.keras import layers, activations

def read_and_preprocess(image_path, image_height, image_width):
    read = tf.io.read_file(image_path)
    image = tf.image.decode_image(read, channels = 3, expand_animations = False)
    image = tf.image.resize(image, (image_height, image_width))
    return tf.cast(image, tf.float32) / 255.0

def aline_sentence(S):
    alined_sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\r\u3000\u2000]+', "", S)
    return alined_sentence

class TransPatch(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, X):
        batch_size = tf.shape(X)[0]
        patches = tf.image.extract_patches(
            images = X,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        num_patches = (X.shape[1] // self.patch_size) * (X.shape[2] // self.patch_size)
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size,
                                       num_patches,
                                       self.patch_size,
                                       self.patch_size,
                                       patch_dim // (self.patch_size * self.patch_size)])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size
        })
        return config

class SelfAttention(layers.Layer):
    """
        args
        D : Dimention of embedding
        D_1 : Dimention of matrix Q and K
        D_2 : Dimention of matrix V
    """
    def __init__(self, D, D_1 = None, D_2 = None, i = None):
        super().__init__()

        self.D = D
        if D_1 is None:
            self.D_1 = self.D
        else:
            self.D_1 = D_1
        if D_2 is None:
            self.D_2 = self.D
        else:
            self.D_2 = D_2
        if i is None:
            self.i = 0
        else:
            self.i = i

        self.W_Q = self.add_weight(shape=(self.D, self.D_1), initializer = 'glorot_uniform', name = f"W_Q_{self.i}")
        self.W_K = self.add_weight(shape=(self.D, self.D_1), initializer = 'glorot_uniform', name = f"W_K_{self.i}")
        self.W_V = self.add_weight(shape=(self.D, self.D_2), initializer = 'glorot_uniform', name = f"W_V_{self.i}")

    def call(self, X):
        Q = tf.matmul(X, self.W_Q)
        K = tf.matmul(X, self.W_K)
        V = tf.matmul(X, self.W_V)

        A = tf.matmul(Q, tf.transpose(K, perm = [0, 2, 1])) / tf.math.sqrt( tf.cast(self.D_1, tf.float32) )
        A = tf.nn.softmax(A, axis = 2)

        return tf.matmul(A, V)

    def get_config(self):
        config = super().get_config()
        config.update({
            'D': self.D,
            'D_1': self.D_1,
            'D_2': self.D_2,
        })
        return config

class MultiHeadSelfAttention(layers.Layer):
    """
        args
        NUM_HEAD : Number of heads
        D : Dimention of embedding
        D_1 : Dimention of matrix Q and K
        D_2 : Dimention of matrix V
    """
    def __init__(self, NUM_HEAD, D, D_1 = None, D_2 = None):
        super().__init__()

        self.NUM_HEAD = NUM_HEAD
        self.heads = list()
        for i in range(self.NUM_HEAD):
            self.heads.append( SelfAttention(D, D_1, D_2, i) )

        self.D = D
        if D_1 is None:
            self.D_1 = self.D
        else:
            self.D_1 = D_1
        if D_2 is None:
            self.D_2 = self.D
        else:
            self.D_2 = D_2

        self.W_O = self.add_weight(shape=(self.D_2 * self.NUM_HEAD, self.D),
                                   initializer = 'glorot_uniform', name = "W_O")

    def call(self, X):
        outputs = list()
        for H in self.heads:
            outputs.append( H(X) )
        X = layers.Concatenate()(outputs)

        return tf.matmul(X, self.W_O)

    def get_config(self):
        config = super().get_config()
        config.update({
            'NUM_HEAD': len(self.heads),
            'D': self.D,
            'D_1': self.D_1,
            'D_2': self.D_2,
        })
        return config

class TrainablePostionEncoding(layers.Layer):
    """
        args
        N : Length of seqence
        D : Dimention of embedding
    """
    def __init__(self, N, D):
        super().__init__()

        self.D = D
        self.N = N
        self.postion_matrix = self.add_weight(shape=(N, D), initializer = 'glorot_uniform',
                                              name = "postion_matrix")

    def call(self, X):
        return X + self.postion_matrix

    def get_config(self):
        config = super().get_config()
        config.update({
            'N': self.N,
            'D': self.D,
        })
        return config

class RowWiseAverage(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return tf.math.reduce_mean(X, axis = 2)

class PatchWiseFlatten(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return tf.map_fn(lambda x : layers.Flatten()(x), X)

class ConcatCLSToken(layers.Layer):
    """
        args
        D : Dimention of embedding
    """
    def __init__(self, D):
        super().__init__()

        self.D = D
        self.cls_token = self.add_weight(shape = (1, 1, self.D), initializer = 'glorot_uniform',
                                         name = "cls_token")

    def call(self, X):
        tmp = tmp = tf.tile(self.cls_token, [tf.shape(X)[0], 1, 1])
        return tf.concat([tmp, X], axis = 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'D': self.D,
        })
        return config

class FeedForward(layers.Layer):
    """
        args
        D : Dimention of embedding
        DROPOUT_RATE : Rate of dropout
    """
    def __init__(self, D, DROPOUT_RATE):
        super().__init__()

        self.D = D
        self.DROPOUT_RATE = DROPOUT_RATE

        self.dense_1 = layers.Dense(units = D * 4, name = "dense_1", kernel_initializer = 'glorot_uniform')
        self.dense_2 = layers.Dense(units = D, name = "dense_2", kernel_initializer = 'glorot_uniform')
        self.dropout_1 = layers.Dropout(rate = DROPOUT_RATE, name = "dropout_1")
        self.dropout_2 = layers.Dropout(rate = DROPOUT_RATE, name = "dropout_2")

    def call(self, X, training = False):
        X_1 = self.dense_1(X)
        X_1 = activations.gelu(X_1)
        X_1 = self.dropout_1(X_1, training = training)
        X_1 = self.dense_2(X_1)
        return self.dropout_2(X_1, training = training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'D': self.D,
            'DROPOUT_RATE': self.dropout_rate,
        })
        return config

class MultiHeadSelfAttentionBlock(layers.Layer):
    """
        args
        NUM_HEAD : Number of heads
        D : Dimention of embedding
        DROPOUT_RATE : Rate of dropout
        D_1 : Dimention of matrix Q and K
        D_2 : Dimention of matrix V
    """
    def __init__(self, NUM_HEAD, D, DROPOUT_RATE = 0.2, D_1 = None, D_2 = None):
        super().__init__()

        self.NUM_HEAD = NUM_HEAD
        self.DROPOUT_RATE = DROPOUT_RATE
        self.D = D
        if D_1 is None:
            self.D_1 = self.D
        else:
            self.D_1 = D_1
        if D_2 is None:
            self.D_2 = self.D
        else:
            self.D_2 = D_2

        self.multi_head_self_attention = MultiHeadSelfAttention(self.NUM_HEAD, self.D, self.D_1, self.D_2)

        self.feed_forward = FeedForward(self.D, self.DROPOUT_RATE)

        self.layer_norm_1 = layers.LayerNormalization(name = "layer_norm_1")
        self.layer_norm_2 = layers.LayerNormalization(name = "layer_norm_2")

    def call(self, X):
        """
        X_1 = self.multi_head_self_attention(X)
        X = self.layer_norm_1(X_1) + X
        X_1 = self.feed_forward(X_1)
        return self.layer_norm_2(X_1) + X
        """
        X_1 = self.multi_head_self_attention(X)
        X = self.layer_norm_1(X_1 + X)
        X_1 = self.feed_forward(X_1)
        return self.layer_norm_2(X_1 + X)

    def get_config(self):
        config = super().get_config()
        config.update({
            'NUM_HEAD': self.NUM_HEAD,
            'D': self.D,
            'DROPOUT_RATE': self.DROPOUT_RATE,
            'D_1': self.D_1,
            'D_2': self.D_2,
        })
        return config
