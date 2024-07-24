import tensorflow as tf
from qssl.config import Config
from qssl.loss.contrastive_pair_loss import Losses
from tensorflow.keras import layers, models, optimizers



class Trainer:
    def __init__(self, siamese_network, pairs_train, labels_train, pairs_test, labels_test):
        self.siamese_network = siamese_network
        self.pairs_train = pairs_train
        self.labels_train = labels_train
        self.pairs_test = pairs_test
        self.labels_test = labels_test

    def train(self, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, learning_rate=Config.LEARNING_RATE):
        tf.get_logger().setLevel('ERROR')

        self.siamese_network.compile(
            loss=Losses.contrastive_pair_loss(),
            optimizer=optimizers.Adam(learning_rate=learning_rate)
        )
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='qssl_hybrid_model.h5', save_weights_only=True, verbose=1)

        history = self.siamese_network.fit(
            [self.pairs_train[:, 0], self.pairs_train[:, 1]], self.labels_train,
            validation_data=([self.pairs_test[:, 0], self.pairs_test[:, 1]], self.labels_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[cp_callback]
        )
        return history
        