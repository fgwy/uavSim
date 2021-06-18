import tensorflow as tf

print(tf.__version__)

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(16, (3,3), padding='same', activation="relu",input_shape=(32, 32, 3)),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),

                tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),

                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),

                tf.keras.layers.Flatten(),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose()
            ]
        )

    def __call__(self, *args, **kwargs):
