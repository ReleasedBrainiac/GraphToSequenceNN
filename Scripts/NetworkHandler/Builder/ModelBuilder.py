import types
from keras.models import Model
from keras.utils import plot_model
from keras.engine import training
from keras import regularizers, activations
from keras import backend as K
from keras.layers import Lambda, concatenate, Dense, Dropout, Input, LSTM, Embedding, Layer, Reshape 
from NetworkHandler.Neighbouring.NeighbourhoodCollector import Neighbourhood as Nhood
from NetworkHandler.KerasSupportMethods.SupportMethods import AssertNotNone, AssertNotNegative, AssertIsKerasTensor

#TODO the unused decoder inputs can be completely removed since i add an external embedding layer.

class ModelBuilder():
    """
    This class allows to easily build a Graph2Sequence neural network model.

    Standard Resources: 
        => 1. https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1765
        => 2. https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        => 3. https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
        => 4. https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/ 
        => 5. https://www.liip.ch/en/blog/sentiment-detection-with-keras-word-embeddings-and-lstm-deep-learning-networks
        => 6. https://medium.com/tensorflow/predicting-the-price-of-wine-with-the-keras-functional-api-and-tensorflow-a95d1c2c1b03

    Recommended resources:
        => 1. https://theailearner.com/2019/01/25/multi-input-and-multi-output-models-in-keras/
        => 2. https://machinelearningmastery.com/keras-functional-api-deep-learning/
    """

    def __init__(self, input_enc_dim: int, edge_dim: int, input_dec_dim: int, input_is_2d:bool =True):
        """
        This constructor collects the necessary dimensions for the GraphEmbedding Network 
        and build the necessary input tensors for the network. 
        They can be accessed by calling 'get_encoder_inputs()' and 'get_decoder_inputs'.
            :param input_enc_dim:int: dimension of the encode inputs
            :param edge_dim:int: dimension of edge look ups
            :param input_dec_dim:int: dimension of the decoder inputs
            :param input_is_2d:bool: this informs the system about the construction of the input shapes
            
        """   
        AssertNotNegative(input_enc_dim), 'Encoder input dim was negative!'
        AssertNotNegative(edge_dim), 'Edge  dim was negative!'
        AssertNotNegative(input_dec_dim), 'Decoder input dim was negative!'

        self.input_enc_dim = input_enc_dim
        self.edge_dim = edge_dim
        self.input_dec_dim = input_dec_dim

        self.input_is_2d = input_is_2d

        self.input_enc_shape = (input_enc_dim,) if (input_is_2d) else (edge_dim,input_enc_dim)
        self.edge_shape = (edge_dim,) if (input_is_2d) else (edge_dim,edge_dim)
        self.input_dec_shape = (input_dec_dim,)

        self.encoder_inputs = self.BuildEncoderInputs()
        self.decoder_inputs = self.BuildDecoderInputs()

    def BuildEncoderInputs(self):
        """
        This function builds the encoder input tensors for the network.
        ATTENTION: 
            Don't call it externally to use it as indirect input for model build!
            If you do so you going to get the 'Disconnected Graph' Error!
            This happens because you are generating NEW Input Tensors instead of getting the exsisting.
            Better use 'get_encoder_inputs()'.
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
            return Input(shape = self.input_dec_shape, name="sentences")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildDecoderInputs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildNeighbourhoodLayer(self, 
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
            template = "An exception of type {0} occurred in [ModelBuilder.BuildNeighbourhoodLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildSingleHopActivation( self,
                                  previous_layer: Layer,
                                  name:str,
                                  hidden_dim: int,
                                  kernel_init: str,
                                  bias_init: str,
                                  act: activations,
                                  kernel_regularizer: regularizers,
                                  activity_regularizer: regularizers,
                                  use_bias: bool,
                                  drop_rate: float):
        """
        This function builds a layer structure for 1 Hop step in the encoder model part.
            :param previous_layer:Layer: the previous layer
            :param name:str: the layer name 
            :param hidden_dim:int: hidden dimension
            :param kernel_init:str: kernel initializer
            :param bias_init:str: bias initializer
            :param act:activations: activation function
            :param kernel_regularizer:regularizers: kernel regularizers
            :param activity_regularizer:regularizers: activity regularizers
            :param use_bias:bool: want result biased
            :param drop_rate:float: dropout percentage
        """
        try:
            dense = Dense(  units=hidden_dim,
                            input_shape = previous_layer.shape,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            activation=act,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer,
                            use_bias=use_bias,
                            name=name+'_time_dist_dense_act')(previous_layer)

            return Dropout(drop_rate, name=name+'_drop')(dense)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildSingleHopActivation]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildDecoderLSTM(   self,
                            inputs:Layer,
                            prev_memory_state, 
                            prev_carry_state,
                            name:str = 'sequence_decoder',
                            training:bool = False,
                            units=0, 
                            act:str ='tanh', 
                            rec_act:str ='hard_sigmoid', 
                            use_bias:bool =True, 
                            kernel_initializer:str ='glorot_uniform', 
                            recurrent_initializer:str ='orthogonal', 
                            bias_initializer:str ='zeros', 
                            unit_forget_bias:bool =True, 
                            kernel_regularizer=None, 
                            recurrent_regularizer=None, 
                            bias_regularizer=None, 
                            activity_regularizer=None, 
                            kernel_constraint=None, 
                            recurrent_constraint=None, 
                            bias_constraint=None, 
                            dropout:float =0.0, 
                            rec_dropout:float =0.0, 
                            implementation:int =1, 
                            return_sequences:bool =True, 
                            return_state:bool =True, 
                            go_backwards:bool =False, 
                            stateful:bool =False, 
                            unroll:bool =False):
        """
        This function is as wrapper for the Keras LSTM with previous states. Its based on the definition in the keras-team gihub repository.
            :Params are the Keras params! [Class Docu -> Standard Resources -> 1.]
        """
        try:
            AssertNotNone(inputs, 'lstm_inputs')
            AssertNotNone(prev_memory_state, 'prev_memory_state')
            AssertNotNone(prev_carry_state, 'prev_carry_state')
            AssertNotNegative(units)

            decoder_lstm = LSTM(name=name,
                                units=units, 
                                activation=act, 
                                recurrent_activation=rec_act, 
                                use_bias=use_bias, 
                                kernel_initializer=kernel_initializer, 
                                recurrent_initializer=recurrent_initializer, 
                                bias_initializer=bias_initializer, 
                                unit_forget_bias=unit_forget_bias, 
                                kernel_regularizer=kernel_regularizer, 
                                recurrent_regularizer=recurrent_regularizer, 
                                bias_regularizer=bias_regularizer, 
                                activity_regularizer=activity_regularizer, 
                                kernel_constraint=kernel_constraint, 
                                recurrent_constraint=recurrent_constraint, 
                                bias_constraint=bias_constraint, 
                                dropout=dropout, 
                                recurrent_dropout=rec_dropout, 
                                implementation=implementation, 
                                return_sequences=return_sequences, 
                                return_state=return_state, 
                                go_backwards=go_backwards, 
                                stateful=stateful, 
                                unroll=unroll)

            return decoder_lstm(inputs=inputs, initial_state=[prev_memory_state, prev_carry_state], training=training)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildDecoderLSTM]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildDecoderPrediction(self, previous_layer:Layer, units:int, act:activations, sentences_dim:int):
        """
        This function build the prediction part of the decoder.
            :param previous_layer:Layer: previous layer (e.g. lstm)
            :param units:int: units for the next dense layer (in this case the word vector features count)
            :param act:activations: the dense layers activations
            :param sentences_dim:int: the sentence embedding features count
        """   
        try:
            encoded_sentences_shape = (sentences_dim,)
            dense_to_word_emb_dim = Dense(units=units, activation=act)(previous_layer)
            denste_to_sentence_emb_dim = Dense(units=1, activation=act)(dense_to_word_emb_dim)
            return Reshape(encoded_sentences_shape)(denste_to_sentence_emb_dim)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildDecoderPrediction]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildGraphEmeddingConcatenation(self, 
                                        forward_layer: Layer =None, 
                                        backward_layer: Layer =None, 
                                        hidden_dim: int =100, 
                                        kernel_init: str ='glorot_uniform',
                                        act: activations = activations.relu,
                                        kernel_regularizer: regularizers =regularizers.l2(0.01),
                                        activity_regularizer: regularizers =regularizers.l1(0.01),):
        """
        This functions builds the node embedding to graph embedding sub model and is element of the encoder structure.
            :param forward_layer:Layer: previous forward layer
            :param backward_layer:Layer: previous backward layer
            :param hidden_dim:int: hidden dimension depends on the embedding dimension e.g. GloVe vector length used. [Default 100]
            :param kernel_init:str: kernel initializer [Default glorot_uniform]
            :param act:activations: activation function [Default relu]
            :param kernel_regularizer:regularizers: kernel regularizers [Default l2(0.01)]
            :param activity_regularizer:regularizers: activity regularizers [Default l1(0.01)] 
        """
        try:
            AssertNotNone(forward_layer, 'forward_layer')
            AssertNotNone(backward_layer, 'backward_layer')
            concat = concatenate([forward_layer,backward_layer], name="fw_bw_concatenation", axis=1)
            hidden_dim = 2* hidden_dim
            concat_act = Dense( hidden_dim, 
                                kernel_initializer=kernel_init,
                                activation=act,
                                kernel_regularizer=kernel_regularizer,
                                activity_regularizer=activity_regularizer,
                                name="concatenation_act")(concat)

            reshape_lambda = lambda x: K.reshape(K.max(x,axis=0), [-1, hidden_dim])

            if(not self.input_is_2d):
                reshape_lambda = lambda x: K.reshape(K.max(x,axis=1), [-1, 1, hidden_dim])

            concat_pool = Lambda(reshape_lambda, name='concat_pool')(concat_act)
            graph_embedding_encoder_states = [concat_pool, concat_pool]


            return [concat_act, graph_embedding_encoder_states]
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildGraphEmeddingConcatenation]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildGraphEmbeddingEncoder(
                            self, 
                            hops: int =1,
                            aggregator: str ='mean',
                            hidden_dim: int =100, 
                            kernel_init: str ='glorot_uniform',
                            bias_init: str ='zeros',
                            act: activations = activations.relu,
                            kernel_regularizer: regularizers =regularizers.l2(0.01),
                            activity_regularizer: regularizers =regularizers.l1(0.01),
                            use_bias: bool =False,
                            drop_rate: float =0.2):
        """
        This function builds the 1st (encoder) part of the Graph2Sequence ANN
            :param hops:int: size of neighbours cover sphere for each node (which neighbours you want to know from your current p.o.v.) [Default 1]
            :param aggregator:str: aggretaor function [Default mean]
            :param hidden_dim:int: hidden dimension depends on the embedding dimension e.g. GloVe vector length used. [Default 100]
            :param kernel_init:str: kernel initializer [Default glorot_uniform]
            :param bias_init:str: bias initializer [Defaul zeros]
            :param act:activations: activation function [Default relu]
            :param kernel_regularizer:regularizers: kernel regularizers [Default l2(0.01)]
            :param activity_regularizer:regularizers: activity regularizers [Default l1(0.01)]
            :param use_bias:bool: want result biased [Default False]
            :param drop_rate:float: dropout percentage [Default 0.2]
        """ 
        try:
            out_shape_lambda = (self.input_enc_dim+self.edge_dim,) if (self.input_is_2d) else (self.edge_dim, self.input_enc_dim+self.edge_dim)
            features_inputs, fw_look_up_inputs, bw_look_up_inputs = self.encoder_inputs
            neighbourhood_func = lambda x: Nhood(x[0], x[1], aggregator=aggregator, is_2d=self.input_is_2d).Execute()

            forward = features_inputs 
            backward = features_inputs 

            for i in range(hops):
                fw_name = ("fw_"+str(i))
                bw_name = ("bw_"+str(i))
                
                forward = self.BuildNeighbourhoodLayer(forward,  fw_look_up_inputs, i,  neighbourhood_func, fw_name, out_shape_lambda)
                forward = self.BuildSingleHopActivation(forward, fw_name, hidden_dim, kernel_init, bias_init, act, kernel_regularizer, activity_regularizer, use_bias, drop_rate)

                backward = self.BuildNeighbourhoodLayer(backward,  bw_look_up_inputs, i,  neighbourhood_func, bw_name, out_shape_lambda)
                backward = self.BuildSingleHopActivation(backward, bw_name, hidden_dim, kernel_init, bias_init, act, kernel_regularizer, activity_regularizer, use_bias, drop_rate)

            return self.BuildGraphEmeddingConcatenation(forward,backward)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildGraphEmbeddingEncoder]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def BuildGraphEmbeddingDecoder( self,
                                    embedding_layer: Embedding,
                                    prev_memory_state: Layer,  
                                    prev_carry_state: Layer,
                                    act:activations = activations.softmax):
        """
        This function builds the 2nd (decoder) part of the Graph2Sequence ANN.
            :param embedding_layer:Embedding: given embedding layer
            :param prev_memory_state:Layer: previous layer mem state
            :param prev_carry_state:Layer: previous layer carry state
            :param act:activations: layers activation function [Default Softmax]
        """
        try:
            AssertIsKerasTensor(embedding_layer)
            AssertIsKerasTensor(prev_memory_state)
            AssertIsKerasTensor(prev_carry_state)

            emb_shape = embedding_layer.shape
            emb_shape_len = len(emb_shape)
            word_emb_dim = int(emb_shape[emb_shape_len-1])
            sentence_dim = int(emb_shape[emb_shape_len-2])
            states_dim = int(prev_memory_state.shape[len(prev_memory_state.shape)-1])

            lstm_decoder_outs, _, _ = self.BuildDecoderLSTM(inputs=embedding_layer, prev_memory_state=prev_memory_state, prev_carry_state=prev_carry_state, units=states_dim)
            return self.BuildDecoderPrediction(previous_layer=lstm_decoder_outs, units=word_emb_dim, act=act, sentences_dim=sentence_dim)
        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.BuildGraphEmbeddingDecoder]. Arguments:\n{1!r}"
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
                     loss:str ='mean_squared_error', 
                     optimizer:str ='rmsprop', 
                     metrics:list =['mae', 'acc']):
        """
        This function compiles the training model.
            :param model:training.Model: the training model
            :param loss:str: name of a loss function [Keras definition]
            :param optimizer:str: name of an optimizer
            :param metrics:list: list of strings of possible metrics [Keras definition]
        """   
        try:
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
            print(model.summary())
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