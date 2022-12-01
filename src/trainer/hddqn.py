from dataclasses import dataclass
from typing import Optional

from src.gym.hcpp import HCPPGym
from src.trainer.ddqn import DDQNTrainer, DDQNTrainerParams
from src.gym.grid import GridGym
from src.base.logger import Logger

from tensorflow.keras.layers import AvgPool2D, Input, Conv2D, Flatten, Dense, Concatenate, MaxPool2D, UpSampling2D
from tensorflow.keras import Model
import tensorflow as tf


@dataclass
class HDDQNTrainerParams(DDQNTrainerParams):
    pass


class HDDQNTrainer(DDQNTrainer):
    def __init__(self, params: HDDQNTrainerParams, gym: HCPPGym, logger: Optional[Logger]):
        super().__init__(params, gym, logger)
        self.params = params

    def create_network(self):
        obs = self.observation_space
        global_map_input = Input(shape=obs["global_map"].shape[1:], dtype=tf.float32)
        local_map_input = Input(shape=obs["local_map"].shape[1:], dtype=tf.float32)
        scalars_input = Input(shape=obs["scalars"].shape[1:], dtype=tf.float32)

        global_map = global_map_input
        local_map = local_map_input
        conv_kernels = self.params.conv_kernels
        hidden_layer_size = self.params.hidden_layer_size
        kernel_size = self.params.conv_kernel_size

        # Feature Extraction
        for _ in range(2):
            global_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(global_map)
            local_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(local_map)

            conv_kernels *= 2
            global_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(global_map)
            local_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(local_map)
            global_map = MaxPool2D(2)(global_map)
            local_map = MaxPool2D(2)(local_map)

        # Global Feature and Scalars mixing
        global_features = tf.reduce_max(tf.reduce_max(global_map, axis=1), axis=1)
        global_features = tf.concat((global_features, scalars_input), axis=1)
        global_features = Dense(hidden_layer_size, activation="relu")(global_features)

        # Local and global Feature Mixing
        _, x, y, _ = local_map.shape
        global_features = tf.reshape(global_features, (-1, 1, 1, hidden_layer_size))
        global_features = tf.repeat(global_features, x, axis=1)
        global_features = tf.repeat(global_features, y, axis=2)
        local_map = tf.concat((local_map, global_features), axis=3)

        # Shared-MLP
        local_map = Conv2D(conv_kernels, 1, activation="relu")(local_map)
        local_map = Conv2D(conv_kernels, 1, activation="relu")(local_map)

        # Feature extraction for landing
        features = tf.reduce_max(tf.reduce_max(local_map, axis=1), axis=1)
        landing_hidden = Dense(hidden_layer_size, activation="relu")(features)
        landing_action = Dense(1, activation=None)(landing_hidden)

        # Upsampling for map
        for _ in range(2):
            local_map = UpSampling2D(2)(local_map)
            conv_kernels /= 2
            local_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(local_map)
            local_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(local_map)

        target = tf.image.crop_to_bounding_box(local_map, 0, 0, *self.gym.params.target_shape) #TODO: check cropping issue (bounding box too big)
        target = Conv2D(1, 1, activation=None)(target)

        output = tf.concat((Flatten()(target), landing_action), axis=1)

        return Model(inputs={"global_map": global_map_input, "local_map": local_map_input, "scalars": scalars_input},
                     outputs=output)