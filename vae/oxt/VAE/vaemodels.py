import numpy as np
import pandas as pd
import os
from keras.models import Model
from keras.layers import Input, Dense, Lambda, LSTM, Flatten, Reshape, Conv2D, Conv2DTranspose, RepeatVector, TimeDistributed
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from sklearn.metrics import f1_score

class VAEModel:
    def __init__(self, input_shape, latent_dim, architecture='dense'):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()
        
    def sampling(self, args):
        mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mu))
        return mu + K.exp(log_var / 2) * epsilon

    def build_encoder(self):
        if self.architecture == 'dense':
            return self.build_dense_encoder()
        elif self.architecture == 'lstm':
            return self.build_lstm_encoder()
        elif self.architecture == 'conv':
            return self.build_conv_encoder()
        else:
            raise ValueError(f'Invalid architecture {self.architecture}')

    def build_decoder(self):
        if self.architecture == 'dense':
            return self.build_dense_decoder()
        elif self.architecture == 'lstm':
            return self.build_lstm_decoder()
        elif self.architecture == 'conv':
            return self.build_conv_decoder()
        else:
            raise ValueError(f'Invalid architecture {self.architecture}')

    def build_dense_encoder(self):
        inputs = Input(shape=self.input_shape)
        x = Flatten()(inputs)
        x = Dense(64, activation='relu')(x)
        mu = Dense(self.latent_dim)(x)
        log_var = Dense(self.latent_dim)(x)
        z = Lambda(self.sampling)([mu, log_var])
        encoder = Model(inputs, [mu, log_var, z], name='encoder')
        return encoder

    def build_lstm_encoder(self):
        inputs = Input(shape=self.input_shape)
        x = LSTM(64, activation='tanh', return_sequences=True)(inputs)
        x = LSTM(32, activation='tanh', return_sequences=True)(x)
        x = LSTM(16, activation='tanh', return_sequences=True)(x)
        x = LSTM(8, activation='tanh', return_sequences=True)(x)
        x = LSTM(4, activation='tanh')(x)
        mu = Dense(self.latent_dim)(x)
        log_var = Dense(self.latent_dim)(x)
        z = Lambda(self.sampling)([mu, log_var])
        encoder = Model(inputs, [mu, log_var, z], name='encoder')
        return encoder

    def build_conv_encoder(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        mu = Dense(self.latent_dim)(x)
        log_var = Dense(self.latent_dim)(x)
        z = Lambda(self.sampling)([mu, log_var])
        encoder = Model(inputs, [mu, log_var, z], name='encoder')
        return encoder

    def build_dense_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(np.prod(self.input_shape), activation='relu')(latent_inputs)
        x = Reshape(self.input_shape)(x)
        outputs = Dense(self.input_shape[1], activation='softmax')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder

    def build_lstm_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = RepeatVector(self.input_shape[0])(latent_inputs)
        x = LSTM(4, activation='tanh', return_sequences=True)(x)
        x = LSTM(8, activation='tanh', return_sequences=True)(x)
        x = LSTM(16, activation='tanh', return_sequences=True)(x)
        x = LSTM(32, activation='tanh', return_sequences=True)(x)
        x = LSTM(64, activation='tanh', return_sequences=True)(x)
        outputs = TimeDistributed(Dense(self.input_shape[1], activation='softmax'))(x)
        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder

    def build_conv_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(18 * 4 * 64, activation='relu')(latent_inputs)
        x = Reshape((18, 4, 64))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder

    def build_vae(self):
        inputs = Input(shape=self.input_shape)
        mu, log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(reconstructed))
        reconstruction_loss *= np.prod(self.input_shape)
        vae_loss = K.mean(kl_loss + reconstruction_loss)
        vae = Model(inputs, reconstructed, name='vae')
        vae.add_loss(vae_loss)
        return vae

    def train(self, X, Y, epochs=100, batch_size=2, verbose=0):
        self.vae.compile(optimizer='adam')
        self.vae.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        if not os.path.exists(f'vae_{self.architecture}_full.png'):
            plot_model(self.vae, to_file=f'vae_{self.architecture}_full.png', show_shapes=True, show_layer_names=True, expand_nested=True)

    def generate_sequences(self, n_samples):
        latent_samples = np.random.normal(size=(n_samples, self.latent_dim))
        generated_sequences = self.decoder.predict(latent_samples)
        nucleotides = ['A', 'C', 'G', 'T']
        sequences = []
        for i in range(n_samples):
            sequence = ''
            for j in range(self.input_shape[0]):
                nucleotide = nucleotides[np.argmax(generated_sequences[i][j])]
                sequence += nucleotide
            if sequence not in sequences:
                sequences.append(sequence)
        sequences_df = pd.DataFrame(sequences, columns=['sequence'])
        return sequences_df
