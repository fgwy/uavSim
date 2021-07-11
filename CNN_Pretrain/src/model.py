import tensorflow as tf

print(tf.__version__)

class Autoencoder_flex(tf.keras.models.Model):

    def __init__(self, num_layers, input_shape):
        super(Autoencoder_flex, self).__init__()
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def call(self, x, *args, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_latentspace(self, x):
        encoded = self.encoder(x)
        return [encoded, encoded.shape]

    def build_encoder(self):
        encoder = tf.keras.Sequential(name='encoder')
        encoder.add(tf.keras.layers.Input(shape=self.input_shape))
        for k in range(self.num_layers):
            encoder.add(tf.keras.layers.Conv2D(filters=(4+4*k), kernel_size=(4, 4), name='{}th-encoding_layer'.format(k), padding='same', activation="elu"))
            encoder.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2))
        return encoder

    def build_decoder(self):
        decoder = tf.keras.Sequential(name='decoder')
        for k in range(self.num_layers):
            decoder.add(tf.keras.layers.Conv2DTranspose(filters=(4+4*(self.num_layers-k)), kernel_size=(4, 4), name='{}th-decoding_layer'.format(k), activation='elu', padding='same'))
            decoder.add(tf.keras.layers.UpSampling2D((2, 2)))

        return decoder


class Autoencoder(tf.keras.models.Model):
    def __init__(self, num_layers, inp_shape):
        super(Autoencoder, self).__init__()
        self.num_layers = num_layers
        self.inp_shape = inp_shape

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=inp_shape),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'),
            tf.keras.layers.Reshape(inp_shape)])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
