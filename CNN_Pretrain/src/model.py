import tensorflow as tf

print(tf.__version__)

class Autoencoder_flex(tf.keras.models.Model):

    def __init__(self, num_layers, inp_shape):
        super(Autoencoder_flex, self).__init__()
        self.num_layers = num_layers
        self.inp_shape = inp_shape
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
        encoder.add(tf.keras.layers.Input(shape=self.inp_shape))
        for k in range(self.num_layers):
            encoder.add(tf.keras.layers.Conv2D(filters=(4+4*k), kernel_size=3, name='{}th-encoding_layer'.format(k), padding='valid', activation="elu"))
            encoder.add(tf.keras.layers.MaxPooling2D((2, 2), strides=1))
        return encoder

    def build_decoder(self):
        decoder = tf.keras.Sequential(name='decoder')
        for k in range(self.num_layers-1):
            decoder.add(tf.keras.layers.Conv2DTranspose(filters=(4+4*(self.num_layers-k)), kernel_size=3, name='{}th-decoding_layer'.format(k), activation='elu', padding='valid'))
            decoder.add(tf.keras.layers.UpSampling2D((2, 2)))
        decoder.add(tf.keras.layers.Conv2DTranspose(filters=self.inp_shape[2], kernel_size=3))

        sh = decoder.output_shape()
        print('decoder shape: {}'.format(sh))
        decoder.add(tf.keras.layers.Conv2DTranspose())

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
            tf.keras.layers.Conv2DTranspose(filters=inp_shape[2], kernel_size=3, activation='sigmoid', padding='same'),
            tf.keras.layers.Reshape(inp_shape)])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
