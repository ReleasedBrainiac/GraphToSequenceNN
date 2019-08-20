import tensorflow as tf
import keras
import numpy as np
from keras import backend as K
from keras import losses, optimizers
from keras.models import Model
from keras.layers import Dense, GRU, Embedding, LSTM, Flatten


class BahdanauAttention(Model):
    def __init__(self, units:int):
        super(BahdanauAttention, self).__init__()
        self._W1 = Dense(units)
        self._W2 = Dense(units)
        self._V = Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = K.expand_dims(query, 1)
        score = self._V(K.tanh(self._W1(values) + self._W2(hidden_with_time_axis)))
        attention_weights = K.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Encoder(Model):

    _rec_init:str='glorot_uniform'
    _return_sequences=True,
    _return_state=True

    def __init__(self, vocab_size:int, embedding_dim:int, enc_units:int, batch_sz:int, use_gru:bool = True):
        super(Encoder, self).__init__()
        self._recurrent = None
        self._use_gru = use_gru
        self._batch_sz = batch_sz
        self._enc_units = enc_units
        self._embedding = Embedding(vocab_size, embedding_dim)

        if self._use_gru:
            self._recurrent = GRU(  self._enc_units,
                                    return_sequences=self._return_sequences,
                                    return_state=self._return_state,
                                    recurrent_initializer=self._rec_init)
        else:
            self._recurrent = LSTM( self._enc_units,
                                    return_sequences=self._return_sequences,
                                    return_state=self._return_state,
                                    recurrent_initializer=self._rec_init)

    def call(self, x, hidden):
        if self._use_gru:
            output, state = self._recurrent(self._embedding(x), initial_state = hidden)
        else:
            output, h, c = self._recurrent(self._embedding(x), initial_state = hidden)
            state = (h, c)
        return output, state

    def initialize_hidden_state(self):
        return K.zeros((self._batch_sz, self._enc_units))

class Decoder(Model):
    _rec_init:str='glorot_uniform'
    _return_sequences=True,
    _return_state=True

    def __init__(self, vocab_size:int, embedding_dim:int, dec_units:int, batch_sz:int):
        super(Decoder, self).__init__()
        self._batch_sz = batch_sz
        self._dec_units = dec_units
        self._embedding = Embedding(vocab_size, embedding_dim)

        if self._use_gru:
            self._recurrent = GRU(  self._dec_units,
                                    return_sequences=self._return_sequences,
                                    return_state=self._return_state,
                                    recurrent_initializer=self._rec_init)
        else:
            self._recurrent = LSTM( self._dec_units,
                                    return_sequences=self._return_sequences,
                                    return_state=self._return_state,
                                    recurrent_initializer=self._rec_init)

        self._out = Dense(vocab_size)
        self._attention = BahdanauAttention(self._dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self._attention(hidden, enc_output)
        embedding_concat_attention = K.concat([K.expand_dims(context_vector, 1), self._embedding(x)], axis=-1)
        output, state = self._recurrent(embedding_concat_attention)
        #output = tf.reshape(output, (-1, output.shape[2]))
        output = Flatten(name='reduce_dimension')(output)

        return self._out(output), state, attention_weights

class Model(object):
    from NetworkHandler.Builder.ModelIterativeBuilder import Encoder

    def __init__(self, word_look_up, start_sign:str, loss_func:losses, optimizer:optimizers, vocab_size:int, embedding_dim:int, batch_sz:int, encoder_units:int, decoder_units:int):
        self._batch_sz:int = batch_sz
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._word_look_up = word_look_up
        self._start_sign = start_sign
        self._encoder_units = encoder_units
        self._decoder_units = decoder_units

    def train_step(self, enc_input, target, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:

            encoder = Encoder(self._vocab_size, self._embedding_dim, self._encoder_units, self._batch_sz)
            enc_output, dec_hidden = encoder(enc_input, enc_hidden)
            dec_input = K.expand_dims([self._word_look_up.word_index[self._start_sign]] * self._batch_sz, 1)

            decoder = Decoder(self._vocab_size, self._embedding_dim, self._decoder_units, self._batch_sz)


            for t in range(1, target.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_func(target[:, t], predictions)

                # using teacher forcing
                dec_input = K.expand_dims(target[:, t], 1)

        batch_loss = (loss / int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss
    
    '''
    def evaluate(self, sentence, max_length_targ:int, max_length_inp:int):
        attention_plot = np.zeros((max_length_targ, max_length_inp))

        sentence = preprocess_sentence(sentence)

        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen=max_length_inp,
                                                            padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self._word_look_up.word_index['<start>']], 0)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                dec_hidden,
                                                                enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self._word_look_up.index_word[predicted_id] + ' '

            if self._word_look_up.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

    def plot_attention(self, attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()

    def translate(self, sentence):
        result, sentence, attention_plot = evaluate(sentence)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        plot_attention(attention_plot, sentence.split(' '), result.split(' '))
    '''