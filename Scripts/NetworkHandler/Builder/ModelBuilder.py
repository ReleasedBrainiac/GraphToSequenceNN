import types
from keras.models import Model
from keras.utils import plot_model
from keras.engine import training
from keras import regularizers, activations
from keras import backend as K
from keras.layers import Lambda, concatenate, Dense, Dropout, Input, LSTM, Embedding, Layer, GlobalMaxPooling1D

from DatasetHandler.ContentSupport import isLambda
from NetworkHandler.Neighbouring.NeighbourhoodCollector import Neighbourhood as Nhood
from NetworkHandler.KerasSupportMethods.SupportMethods import AssertNotNone, AssertNotNegative, AssertIsKerasTensor

#TODO finish docu
#TODO ~> Resource for MA and Code ~> https://stackoverflow.com/questions/32771786/predictions-using-a-keras-recurrent-neural-network-accuracy-is-always-1-0/32788454#32788454 
#TODO ~> Best LSTM resources ~> https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
#TODO ~> 2nd best LSTM resource ~> https://adventuresinmachinelearning.com/keras-lstm-tutorial/

class ModelBuilder():

    def __init__(self, input_enc_dim: int, edge_dim: int, input_dec_dim: int):
        """
        This constructor collects the necessary dimensions for the GraphEmbedding Network 
        and build the necessary input tensors for the network. 
        They can be accessed by calling 'get_encoder_inputs()' and 'get_decoder_inputs'.
            :param input_enc_dim:int: dimension of the encode inputs
            :param edge_dim:int: dimension of edge look ups
            :param input_dec_dim:int: dimension of the decoder inputs
        """   
        AssertNotNegative(input_enc_dim), 'Encoder input dim was negative!'
        AssertNotNegative(edge_dim), 'Edge  dim was negative!'
        AssertNotNegative(input_dec_dim), 'Decoder input dim was negative!'

        self.input_enc_dim = input_enc_dim
        self.edge_dim = edge_dim
        self.input_dec_dim = input_dec_dim

        self.input_enc_shape = (input_enc_dim,)
        self.edge_shape = (edge_dim,)
        self.input_dec_shape = (input_dec_dim,)

        self.encoder_inputs = self.BuildEncoderInputs()
        self.decoder_inputs = self.BuildDecoderInputs()

    def BuildEncoderInputs(self):
        """
        This function build the encoder input tensors for the network.
        ATTENTION: 
            Don't call it externally to use it as indirect input for model build!
            If you do so you going to get the 'Disconnected Graph' Error!
            This happens because you are generating NEW Input Tensors instead of getting the exsisting.
            Better use 'get_encoder_inputs()'.
        """   
        features = Input(shape = self.input_enc_shape, name="features")
        foward_look_up = Input(shape = self.edge_shape, name="fw_neighbourhood")
        backward_look_up =  Input(shape = self.edge_shape, name="bw_neighbourhood")
        return [features, foward_look_up, backward_look_up]

    def BuildDecoderInputs(self):
        """
        This function build the decoder input tensors for the network.
        ATTENTION: 
            Don't call it externally to use it as indirect input for model build!
            If you do so you going to get the 'Disconnected Graph' Error!
            This happens because you are generating NEW Input Tensors instead of getting the exsisting.
            Better use 'get_decoder_inputs()'.
        """   
        return Input(shape = self.input_dec_shape, name="sentences")

    def BuildNeighbourhoodLayer(self, 
                                features, 
                                look_up, 
                                hop:int, 
                                hood_func: types.LambdaType, 
                                layer_name: str, 
                                out_shape: list):
        """
        This function build a neighbourhood collecting keras lambda layer.
            :param features: 
            :param look_up: 
            :param hop:int: 
            :param hood_func:types.LambdaType: 
            :param layer_name:str: 
            :param out_shape:list: 
        """

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
    
        x = Dense(  units=hidden_dim,
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    activation=act,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
                    use_bias=use_bias,
                    name=name+'_dense_act')(previous_layer)
        
        return Dropout(drop_rate, name=name+'_drop')(x)

    def BuildDecoderLSTM(   self,
                            inputs:Layer,
                            prev_memory_state, 
                            prev_carry_state,
                            name:str = 'LSTMDecoder',
                            training:bool = True,
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
        Resources: 
            => https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1765
            => https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
            => https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
            => https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/ 
            => https://www.liip.ch/en/blog/sentiment-detection-with-keras-word-embeddings-and-lstm-deep-learning-networks
            => https://medium.com/tensorflow/predicting-the-price-of-wine-with-the-keras-functional-api-and-tensorflow-a95d1c2c1b03
        
        Params are the Keras params!
        """
        AssertNotNone(inputs, 'lstm_inputs')
        AssertNotNone(prev_memory_state, 'prev_memory_state')
        AssertNotNone(prev_carry_state, 'prev_carry_state')
        AssertNotNegative(units)

        outputs, state_h, state_c = LSTM(   units=units, 
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
                                            unroll=unroll)(inputs=inputs, initial_state=[prev_memory_state, prev_carry_state], training=training)

        return [outputs, state_h, state_c]
    
    def BuildGraphEmeddingConcatenation(self, 
                                        forward_layer: Layer =None, 
                                        backward_layer: Layer =None, 
                                        hidden_dim: int =100, 
                                        kernel_init: str ='glorot_uniform',
                                        act: activations = activations.relu,
                                        kernel_regularizer: regularizers =regularizers.l2(0.01),
                                        activity_regularizer: regularizers =regularizers.l1(0.01),):
        
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



        #TODO the following 2 lines are according to the implementation idea in https://github.com/IBM/Graph2Seq/blob/master/main/model.py line 204
        concat_pool = Lambda(lambda x: K.reshape(K.max(x,axis=0), [-1, hidden_dim]))(concat_act)
        print('concat_pool', concat_pool)
        print('concat_pool', type(concat_pool))


        #pool_act = Dense( hidden_dim, name="pool_act")(concat_pool)
        #print('pool_act', pool_act)
        #print('pool_act', type(pool_act))

        #graph_embedding_encoder_states = [concat_pool, concat_pool]
        graph_embedding_encoder_states = [concat_pool, concat_pool]

        print('graph_embedding_encoder_states', graph_embedding_encoder_states)
        print('graph_embedding_encoder_states', type(graph_embedding_encoder_states))

        #TODO [encoder_out, embedding: [hidden state, cell state]]
        return [concat_act, graph_embedding_encoder_states]

    def GraphEmbeddingEncoderBuild(
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
        docstring here
            :param self: 
            :param hops:int=1: 
            :param aggregator:str='mean': 
            :param hidden_dim:int=100: 
            :param kernel_init:str='glorot_uniform': 
            :param bias_init:str='zeros': 
            :param act:activations=activations.relu: 
            :param kernel_regularizer:regularizers=regularizers.l2(0.01: 
        """ 
        
        out_shape_lambda = (self.input_enc_dim+self.edge_dim,)
        print('lambda_shape: ', out_shape_lambda)
        features_inputs, fw_look_up_inputs, bw_look_up_inputs = self.encoder_inputs
        neighbourhood_func = lambda x: Nhood(x[0], x[1], aggregator=aggregator).Execute()
        
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

    def GraphEmbeddingDecoderBuild( self,
                                    embedding_layer: Embedding,
                                    prev_memory_state,  
                                    prev_carry_state, 
                                    num_dec_tokens:int, 
                                    units:int =200,
                                    act:activations = activations.softmax):
        AssertNotNegative(units)
        AssertNotNone(embedding_layer, 'embedding_layer')
        AssertNotNegative(num_dec_tokens)
        AssertNotNone(prev_memory_state, 'encoder_memory_states')
        AssertNotNone(prev_carry_state, 'encoder_carry_states')


        lstm_decoder_outs, _, _ = self.BuildDecoderLSTM(inputs=embedding_layer, prev_memory_state=prev_memory_state, prev_carry_state=prev_carry_state, units=units)
        return Dense(units=100, activation=act)(lstm_decoder_outs)

    def MakeModel(self, layers: Layer):
        """
        This function creates the model by given inputs.
            :param inputs:list=None: list of inputs
            :param layers:Layer=None: layer structure
        """
        inputs = self.get_inputs()
        assert isinstance(inputs, list) , 'The given inputs is no list!'
        AssertNotNone(layers, 'layers')
        return Model(inputs=inputs, outputs=layers)

    #def CompileModel(self, )

    def Summary(self, model: training.Model):
        """
        This function prints the summary of a model.
            :param model:training.Model: a build model
        """   
        AssertNotNone(model, 'plotting_tensor'), 'Plotting model was None!'
        print(model.summary())

    def Plot(self, model: training.Model, file_name: str, show_shapes:bool =True):
        """
        This function plot and store a given model to desired file.
            :param model:Model: keras model
            :param file_name:str: name of the image file
            :param show_shapes:bool =True: show layer shape in the graph
        """ 
        AssertNotNone(model, 'plotting_tensor'), 'Plotting model was None!'
        AssertNotNone(file_name, 'name_plot_file'), 'Plot file name was None!'
        plot_model(model, to_file=file_name, show_shapes=show_shapes)

    def get_encoder_inputs(self):
        """
        This getter returns the encoder inputs.
        """   
        return self.encoder_inputs

    def get_decoder_inputs(self):
        """
        This getter returns the decoder inputs.
        """   
        return self.decoder_inputs

    def get_inputs(self):
        """
        This getter returns the encoder and decoder inputs in exactly this order [encoder, decoder].
        """   
        AssertNotNone(self.encoder_inputs, 'encoder inputs')
        AssertNotNone(self.decoder_inputs, 'decoder inputs')
        return [self.encoder_inputs[0], self.encoder_inputs[1], self.encoder_inputs[2], self.decoder_inputs]