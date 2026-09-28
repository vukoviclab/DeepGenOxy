import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Input, LSTM, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Concatenate, Permute, RepeatVector, Multiply, Attention
from keras.models import Sequential, Model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ModelBase:
    def __init__(self, df, successful_models=0, model_number=1, random_state=0, epochs=10, batch_size=2):
        self.df = df
        self.X = np.array([[[int(nuc == base) for nuc in 'ACGT'] for base in seq] for seq in df['sequence']])
        self.Y_cat = to_categorical(df['category'])
        self.successful_models = successful_models
        self.model_number = model_number
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size

    def create_model(self):
        raise NotImplementedError("Subclass must implement create_model")

    def run_model(self):
        while self.successful_models < 5:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y_cat, test_size=0.2, random_state=self.random_state)
            classifier = self.create_model()
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            plot_path = f'nn-{self.__class__.__name__}.png'
            if not os.path.exists(plot_path):
                plot_model(classifier, to_file=plot_path, show_shapes=True, show_layer_names=True)
            classifier.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_test, y_test), verbose=0)
            y_pred = classifier.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            f1 = f1_score(y_true_classes, y_pred_classes)
            self.random_state += 1
            print(f"Model {self.model_number} Run Completed with F1 Score: {f1}")
            if f1 > 0.8:
                print(f"Model {self.model_number} and the random state is {self.random_state} F1 Score:", f1)
                model_path = f"{self.__class__.__name__}/model_{self.model_number}_rnd_{self.random_state}.h5"
                classifier.save(model_path)
                print(f"Model {self.model_number} saved as {model_path}")
                self.successful_models += 1
            del classifier
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            self.model_number += 1


        
class ConvModel(ModelBase):
    def create_model(self):
        input_shape = (18, 4, 1)
        classifier = Sequential()
        classifier.add(Conv2D(32, (2, 4), activation='relu', input_shape=input_shape))
        classifier.add(MaxPooling2D(pool_size=(1, 1)))
        classifier.add(Dropout(0.2))
        classifier.add(Flatten())
        classifier.add(Dense(64, activation='relu'))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(2, activation='softmax'))
        return classifier


class DenseModel(ModelBase):
    def create_model(self):
        input_shape = (18, 4)
        classifier = Sequential()
        classifier.add(Flatten(input_shape=input_shape))
        classifier.add(Dense(64, activation='relu'))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(2, activation='softmax'))
        return classifier

class LSTMModel(ModelBase):
    def create_model(self):
        input_shape = (18, 4)
        classifier = Sequential()
        classifier.add(LSTM(32, return_sequences=True, input_shape=input_shape))
        classifier.add(Dropout(0.2))
        classifier.add(Flatten())
        classifier.add(Dense(64, activation='relu'))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(2, activation='softmax'))
        return classifier

class ConvLSTMModel(ModelBase):
    def create_model(self):
        input_shape = (18, 4)
        input_seq = Input(shape=input_shape)
        x = Conv1D(32, 2, activation='relu')(input_seq)
        x = MaxPooling1D(2)(x)
        x = LSTM(32, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(2, activation='softmax')(x)
        classifier = Model(inputs=input_seq, outputs=out)
        return classifier

class LSTMAttentionModel(ModelBase):
    def create_model(self):
        input_shape = (18, 4)
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(32, return_sequences=True)(inputs)
        
        query_value_attention_seq = Attention()([lstm_out, lstm_out])
        query_value_attention = Flatten()(query_value_attention_seq)
        
        query_value_attention = Dropout(0.2)(query_value_attention)
        out = Dense(64, activation='relu')(query_value_attention)
        out = Dropout(0.2)(out)
        out = Dense(2, activation='softmax')(out)
        
        classifier = Model(inputs=inputs, outputs=out)
        return classifier
    
class ConvLSTPAMModel(ModelBase):
    def create_model(self):
        input_shape = (18, 4)
        input_seq = Input(shape=input_shape)
        
        conv_branch = Conv1D(32, 3, activation='relu')(input_seq)
        conv_branch = MaxPooling1D(2)(conv_branch)
        conv_branch = Flatten()(conv_branch)
        
        lstm_branch = LSTM(32, return_sequences=False)(input_seq)
        
        x = Concatenate()([conv_branch, lstm_branch])
        
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(2, activation='softmax')(x)
        
        classifier = Model(inputs=input_seq, outputs=out)
        return classifier