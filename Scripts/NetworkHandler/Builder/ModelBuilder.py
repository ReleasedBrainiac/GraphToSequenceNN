import types
import tensorflow as tf
from keras.models import Model
from keras.utils import plot_model
from keras.engine import training
from keras import regularizers, activations
from keras import backend as K
from keras.optimizers import RMSprop, Adam, Nadam, Adagrad, Adadelta
from keras.layers import Lambda, concatenate, Dense, Dropout, Input, LSTM, Embedding, Layer, GlobalMaxPooling1D, RepeatVector, Activation, multiply, add, Flatten, BatchNormalization, TimeDistributed
from NetworkHandler.Neighbouring.NeighbourhoodCollector import Neighbourhood as Nhood
from NetworkHandler.KerasSupportMethods.SupportMethods import AssertNotNone, AssertNotNegative, AssertIsKerasTensor

class ModelBuilder:
    """
    This class allows to easily build a Graph2Sequence neural network model.

    Standard Resources: 
        => 1. https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1765
        => 2. https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        => 3. https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
        => 4. https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/ 
        => 5. https://www.liip.ch/en/blog/sentiment-detection-with-keras-word-embeddings-and-lstm-deep-learning-networks
        => 6. https://medium.com/tensorflow/predicting-the-price-of-wine-with-the-keras-functional-api-and-tensorflow-a95d1c2c1b03
        => 7. https://stackoverflow.com/questions/51749404/how-to-connect-lstm-layers-in-keras-repeatvector-or-return-sequence-true

    Recommended resources:
        => 1. https://theailearner.com/2019/01/25/multi-input-and-multi-output-models-in-keras/
        => 2. https://machinelearningmastery.com/keras-functional-api-deep-learning/
    """

    def __init__(self, input_enc_dim: int, edge_dim: int, input_dec_dim: int, batch_size:int = 1, input_is_2d:bool =False):
        """
        This constructor collects the necessary dimensions for the GraphEmbedding Network 
        and build the necessary input tensors for the network. 
        They can be accessed by calling 'get_encoder_inputs()' and 'get_decoder_inputs'.
            :param input_enc_dim:int: dimension of the encode inputs
            :param edge_dim:int: dimension of edge look ups
            :param input_dec_dim:int: dimension of the decoder inputs
            :param batch_size:int: batch size
            :param input_is_2d:bool: this informs the system about the construction of the input shapes
        """   
        AssertNotNegative(input_enc_dim), 'Encoder input dim was negative!'
        AssertNotNegative(edge_dim), 'Edge  dim was negative!'
        AssertNotNegative(input_dec_dim), 'Decoder input dim was negative!'
        AssertNotNegative(batch_size), 'Batch size was negative!'

        self.input_enc_dim = input_enc_dim
        self.edge_dim = edge_dim
        self.input_dec_dim = input_dec_dim
        self.batch_size = batch_size

        self.input_is_2d = input_is_2d

        self.input_enc_shape = (input_enc_dim,) if (input_is_2d) else (edge_dim,input_enc_dim)
        self.edge_shape = (edge_dim,) if (input_is_2d) else (edge_dim,edge_dim)
        self.input_dec_shape = (input_dec_dim,) if (input_dec_dim > 0) else (1,)

        self.encoder_inputs = self.BuildEncoderInputs()
        self.decoder_inputs = self.BuildDecoderInputs()

    def BuildEncoderInputs(self):
        """
        This function builds the encoder input tensors for the network.
        ATTENTION: 
            Don't call it externally to use it as indirect input for model build!
            If you do so you going to get the 'Disconnected Graph' Error!
            This happens because you are generating NEW Input Tensors instead of getting the exsisting.
        """   
        try:
            features = Input(shape = self.input_enc_shape, name="features")
            foward_look_up = Input(shape = self.edge_shape, name="fw_neighbourhood")
            backward_look_up =  Input(shape = self.edge_shape, name="bw_neighbourhood")
            return [features, foward_look_up, backward_look_up]            
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildEncoderInputs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildDecoderInputs(self):
        """
        This function builds the decoder input tensors for the network.
        ATTENTION: 
            Don't call it externally to use it as indirect input for model build!
            If you do so you going to get the 'Disconnected Graph' Error!
            This happens because you are generating NEW Input Tensors instead of getting the exsisting.
            Better use 'get_decoder_inputs()'.
        """   
        try:
            return Input(shape = self.input_dec_shape, name="words")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildDecoderInputs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def NhoodLambdaLayer(self, 
                                features, 
                                look_up, 
                                hop:int, 
                                hood_func: types.LambdaType, 
                                layer_name: str, 
                                out_shape: list):
        """
        This function builds a neighbourhood collecting keras lambda layer.
            :param features: 2D tensor with rows of vectoriced words 
            :param look_up: 2D tensor with neighbourhood description for foward,backward or both in one
            :param hop:int: hop position in the network model structure
            :param hood_func:types.LambdaType: a lambda neighbourhood function which matches the input structure
            :param layer_name:str: name of the layer (remind an extension will be added)
            :param out_shape:list: shape of the output
        """
        try:
            name = layer_name if layer_name is not None else ''
            name_ext = '_lambda_init' if hop == 0 else '_lambda_step'

            AssertIsKerasTensor(features)
            AssertIsKerasTensor(look_up)
            AssertNotNone(features, 'features'), 'Input tensor for features was None!'
            AssertNotNone(look_up, 'look_up'), 'Input tensor for look_up was None!'
            AssertNotNone(out_shape, 'out_shape'), 'Input for out_shape was None!'
            AssertNotNegative(hop), 'input dimension was negative or none!'
            dataset = [features, look_up]
            return Lambda(hood_func, output_shape=out_shape, name=name+name_ext)(dataset)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.NhoodLambdaLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildDecoderPrediction(self, previous_layer:Layer, act:activations = activations.softmax):
        """
        This function build the prediction part of the decoder.
            :param previous_layer:Layer: previous layer (e.g. lstm)
            :param act:activations: the dense layers activations
        """   
        try:
            flatten = Flatten(name='reduce_dimension')(previous_layer)
            return Dense(units=self.input_dec_dim, activation=act, name='dense_predict')(flatten)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildDecoderPrediction]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildGraphEmeddingConcatenation(self, 
                                        forward_layer: Layer =None, 
                                        backward_layer: Layer =None, 
                                        hidden_dim: int =100, 
                                        kernel_init: str ='glorot_uniform',
                                        act: activations = activations.softmax,
                                        kernel_regularizer: regularizers =regularizers.l2(0.01),
                                        activity_regularizer: regularizers =regularizers.l1(0.01)
                                        ):
        """
        This functions builds the node embedding to graph embedding sub model and is element of the encoder structure.
            :param forward_layer:Layer: previous forward layer
            :param backward_layer:Layer: previous backward layer
            :param hidden_dim:int: hidden dimension depends on the embedding dimension e.g. GloVe vector length used. [Default 100]
            :param kernel_init:str: kernel initializer [Default glorot_uniform]
            :param act:activations: activation function [Default relu]
            #:param kernel_regularizer:regularizers: kernel regularizers [Default l2(0.01)]
            #:param activity_regularizer:regularizers: activity regularizers [Default l1(0.01)] 
        """
        try:
            AssertNotNone(forward_layer, 'forward_layer')
            AssertNotNone(backward_layer, 'backward_layer')
            concat = concatenate([forward_layer,backward_layer], name="fw_bw_concatenation", axis=1)
            #hidden_dim = 2* hidden_dim

            
            hidden = Dense( hidden_dim, 
                            kernel_initializer=kernel_init,
                            activation=act,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer,
                            name="concatenation_act")(concat)
            
            
            #hidden = Dense(units=hidden_dim, activation=act, name='concatenation_act')(concat)

            concat_pool = None
            if(not self.input_is_2d):
                concat_pool = GlobalMaxPooling1D(data_format='channels_last', name='concat_max_pooling')(hidden)
            else: 
                concat_pool = Lambda(lambda x: K.reshape(K.max(x,axis=0), (-1, hidden_dim)), name='concat_max_pool')(hidden)

            graph_embedding_state_h = concat_pool
            graph_embedding_state_c = concat_pool

            return [hidden, graph_embedding_state_h, graph_embedding_state_c]
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildGraphEmeddingConcatenation]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildGraphEmbeddingLayers(
                            self, 
                            hops: int =1,
                            aggregator: str ='mean',
                            hidden_dim: int =100, 
                            act: activations = activations.relu):
        """
        This function builds the 1st (encoder) part of the Graph2Sequence ANN
            :param hops:int: size of neighbours cover sphere for each node (which neighbours you want to know from your current p.o.v.) [Default 1]
            :param aggregator:str: aggretaor function [Default mean]
            :param hidden_dim:int: hidden dimension depends on the embedding dimension e.g. GloVe vector length used. [Default 100]
            :param act:activations: activation function [Default relu]
        """ 
        try:
            extension = "_dense_act"
            out_shape_lambda = (self.input_enc_dim+self.edge_dim,) if (self.input_is_2d) else (self.edge_dim, self.input_enc_dim+self.edge_dim)
            features_inputs, fw_look_up_inputs, bw_look_up_inputs = self.encoder_inputs
            neighbourhood_func = lambda x: Nhood(x[0], x[1], aggregator=aggregator, is_2d=self.input_is_2d).Execute()

            forward = features_inputs 
            backward = features_inputs 

            for i in range(hops):
                fw_name = ("fw_"+str(i))
                bw_name = ("bw_"+str(i))
                
                
                forward = self.NhoodLambdaLayer(forward,  fw_look_up_inputs, i,  neighbourhood_func, fw_name, out_shape_lambda)
                forward = Dense(units=hidden_dim, activation=act, name=fw_name+extension)(forward)

                backward = self.NhoodLambdaLayer(backward,  bw_look_up_inputs, i,  neighbourhood_func, bw_name, out_shape_lambda)
                backward = Dense(units=hidden_dim, activation=act, name=bw_name+extension)(backward)

            return self.BuildGraphEmeddingConcatenation(forward,backward, hidden_dim=hidden_dim)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildGraphEmbeddingLayers]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    ######################################################################################

    def BuildStatePassingEncoder(   self,
                                    sequence_embedding: Embedding,
                                    graph_embedding: Layer,
                                    prev_memory_state: Layer,  
                                    prev_carry_state: Layer,
                                    act = activations.relu):
        try:
            units = int(prev_memory_state.shape[len(prev_memory_state.shape)-1])
            encoder_out, enc_h, enc_c = LSTM(   name="encoder_lstm", 
                                                units=units, 
                                                batch_size=self.batch_size, 
                                                activation=act, 
                                                return_sequences =True, 
                                                return_state =True)(inputs=graph_embedding, initial_state=[prev_memory_state, prev_carry_state])

            #This part is based on https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/ ~> Recursive Model B
            #Encoder Attention
            attention_out, att_weights = self.BuildBahdanauAttentionPipe(units, encoder_out, enc_h)
            attention_reshaped = Lambda(lambda q: K.expand_dims(q, axis=1), name="attention_reshape")(attention_out)

            #Recursive style Embedding
            embedding_lstm = LSTM(attention_reshaped.shape[-1].value, batch_size=self.batch_size, return_sequences=True, name="attention_activate")(sequence_embedding)

            print("batches: ", self.batch_size)
            print("attention: ", attention_reshaped.shape)
            print("embedding: ", embedding_lstm.shape)

            stated_att_encoder = concatenate([attention_reshaped,embedding_lstm], name="att_emb_concatenation", axis=-1)

            print("result: ", stated_att_encoder.shape)

            return units, stated_att_encoder, enc_h, enc_c, att_weights
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildStatePassingEncoder]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildRecursiveEncoder(  self,
                                sequence_lenght:int,
                                sequence_embedding: Embedding,
                                graph_embedding: Layer,
                                prev_memory_state: Layer,  
                                prev_carry_state: Layer,
                                act = activations.relu):
        """
        This implemenation is based on https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/ ~> Recursive Model B
        """
        try: 
            units = int(prev_memory_state.shape[len(prev_memory_state.shape)-1])
            encoder_out = LSTM( name="encoder_lstm", 
                                units=units, 
                                batch_size=self.batch_size, 
                                activation=act)(inputs=graph_embedding, initial_state=[prev_memory_state, prev_carry_state])

            repeated_graph_embedding = RepeatVector(sequence_lenght)(encoder_out)

            embedding_lstm = LSTM(  units=encoder_out.shape[-1].value, 
                                    batch_size=self.batch_size,
                                    return_sequences=True, 
                                    name="attention_activate")(sequence_embedding)

            encoder = concatenate([repeated_graph_embedding, embedding_lstm], name="repeat_emb_concatenation", axis=-1)
            return units, encoder
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildRecursiveEncoder]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    ######################################################################################

    def BuildBahdanauAttentionPipe(self, units:int, sample_outs:Layer, sample_state:Layer):
        """
        This method implements the functionality of the "BahdanauAttention". 
        It is an abstaction from: 
        1. https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention example and 
        2. https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/ example
            :param units:int: dense units for the given sample out and hidden
            :param sample_outs:Layer: encoder or decoder sample outs
            :param sample_state:Layer: encoder or decoder hidden states
        """
        try:
            #Example Res. 2: 
            # time_steps = sample_outs.shape[1]
            # hidden_with_time_axis = K.repeat(sample_state, time_steps)
            hidden_with_time_axis = Lambda(lambda q: K.expand_dims(q, axis=1), name="expand_time_axis")(sample_state)
            
            #Example Res. 1: outs = Dense(units, name="dense_sample_outs")(sample_outs)
            outs = TimeDistributed(Dense(units, name="dense_sample_outs"))(sample_outs)
            hidd = Dense(units, name="dense_hidden_outs")(hidden_with_time_axis)

            #Example Res. 1: score = Dense(1, activation="tanh", name="dense_score")(add[outs, hidd])
            score = Activation("tanh", name="dense_score")(outs + hidd)
            attention_weights = Activation("softmax", name="softmax_bahdanau")(score)

            context_vector = Lambda(lambda z: K.sum(z, axis=1), name="lambda_context_vector")(multiply([attention_weights, sample_outs]))
            return context_vector, attention_weights
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildBahdanauAttentionPipe]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildDecoder(   self,
                        units:int,
                        encoder: Layer,
                        prev_memory_state: Layer,  
                        prev_carry_state: Layer,
                        act = activations.relu,
                        drop_rate:float = 0.5,
                        use_batch_norm:bool=True):
        """
        This function builds the decoder of the Graph2Sequence ANN.
            :param units:count of decoder cells
            :param encoder:Layer: given encoder out layer
            :param prev_memory_state:Layer: previous layer mem state
            :param prev_carry_state:Layer: previous layer carry state
            :param act: layers activation function [Default Softmax]
            :param drop_rate:float: dropout percentage
        """
        try:
            decoder = LSTM( name="decoder_lstm", 
                            units=units, 
                            batch_size=self.batch_size, 
                            activation=act,
                            return_sequences =True, 
                            return_state =True)
                                            
            if (prev_memory_state != None and prev_carry_state != None):
                lstm_decoder_outs, _, _ = decoder(inputs=encoder, initial_state=[prev_memory_state, prev_carry_state], training=True)
            else:
                lstm_decoder_outs, _, _ = decoder(inputs=encoder, training=True)

            if use_batch_norm:
                lstm_decoder_outs = BatchNormalization()(lstm_decoder_outs)

            return Dropout(drop_rate, name='decoder_drop')(lstm_decoder_outs)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildDecoder]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def MakeModel(self, layers: Layer):
        """
        This function creates the model by given inputs.
            :param layers:Layer: structure that defines your model
        """
        try:
            inputs = self.get_inputs()
            AssertNotNone(inputs, 'inputs')
            AssertNotNone(layers, 'layers')
            return Model(inputs=inputs, outputs=layers)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.MakeModel]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def CompileModel(self, 
                     model:training.Model, 
                     loss:str='categorical_crossentropy', 
                     optimizer:str='adam', 
                     metrics:list=['acc'],
                     clipvalue:float=20.0,
                     learn_rate:float=0.001):
        """
        This function compiles the training model.
            :param model:training.Model: the training model
            :param loss:str: name of a loss function [Keras definition]
            :param optimizer: name of an optimizer
            :param metrics:list: list of strings of possible metrics [Keras definition]
            :param clipvalue:float: control gradient clipping
            :param learn_rate:float: optimizer learning rate
        """   
        try:
            print("Compile Model!")

            optimizer = self.get_optimizer(name=optimizer, clipvalue=clipvalue, learn_rate=learn_rate)

            model.compile(  loss=loss,
                            optimizer=optimizer,
                            metrics=metrics)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.CompileModel]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def Summary(self, model:training.Model):
        """
        This function prints the summary of a model.
            :param model:training.Model: a build model
        """
        try:
            AssertNotNone(model, 'plotting_tensor'), 'Plotting model was None!'
            print(model.summary(line_length=200))
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.Summary]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def Plot(self, model:training.Model, file_name:str, show_shapes:bool =True):
        """
        This function plots and store a given model to desired file.
            :param model:Model: keras model
            :param file_name:str: name of the image file
            :param show_shapes:bool =True: show layer shape in the graph
        """
        try:
            AssertNotNone(model, 'plotting_tensor'), 'Plotting model was None!'
            AssertNotNone(file_name, 'name_plot_file'), 'Plot file name was None!'
            plot_model(model, to_file=file_name, show_shapes=show_shapes)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.Plot]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def get_optimizer(self, name:str,  clipvalue:float=20.0, learn_rate:float=0.001, amsgrad:bool=False, decay:float=0.0):
        try:
            if name == 'rmsprop':
                return RMSprop(lr=learn_rate, clipvalue=clipvalue)

            if name == 'adam':
                return Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=amsgrad, clipvalue=clipvalue)

            if name == 'Adagrad':
                return Adagrad(lr=learn_rate, epsilon=None, decay=decay, clipvalue=clipvalue)

            if name == 'Adadelta':
                return Adadelta(lr=learn_rate, rho=0.95, epsilon=None, decay=decay, clipvalue=clipvalue)

            if name == 'Nadam':
                return Nadam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipvalue=clipvalue)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.get_optimizer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def get_encoder_inputs(self):
        """
        This getter returns the encoder inputs.
        """   
        try:
            return self.encoder_inputs
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.get_encoder_inputs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def get_decoder_inputs(self):
        """
        This getter returns the decoder inputs.
        """   
        try:
            return self.decoder_inputs
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.get_decoder_inputs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def get_inputs(self):
        """
        This getter returns the encoder and decoder inputs in exactly this order [encoder, decoder].
        """   
        try:
            AssertNotNone(self.encoder_inputs, 'encoder inputs')
            AssertNotNone(self.decoder_inputs, 'decoder inputs')
            return [self.encoder_inputs[0], self.encoder_inputs[1], self.encoder_inputs[2], self.decoder_inputs]
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.get_inputs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 