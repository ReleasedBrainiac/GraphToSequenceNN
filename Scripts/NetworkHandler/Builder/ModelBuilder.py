import types
from keras.models import Model
from keras.utils import plot_model
from keras.engine import training
from keras import regularizers, activations
from keras.layers import Lambda, concatenate, Dense, Dropout, Input, LSTM, Embedding, Layer

from DatasetHandler.ContentSupport import isLambda
from NetworkHandler.Neighbouring.NeighbourhoodCollector import Neighbourhood as Nhood
from NetworkHandler.KerasSupportMethods.SupportMethods import AssertNotNone, AssertNotNegative, IsKerasTensor

#TODO finish docu

class ModelBuilder():

    def __init__(self, input_dim: tuple, edge_dim: tuple,):
        AssertNotNone(input_dim, 'input_dim'), 'Input tuple was None!'
        self.input_dim = input_dim
        AssertNotNone(edge_dim, 'edge_dim'), 'Edge tuple was None!'
        self.edge_dim = edge_dim

        self.inputs = self.BuildInputs()

    def BuildInputs(self):
        features = Input(shape = self.input_dim, name="features")
        foward_look_up = Input(shape = self.edge_dim, name="fw_neighbourhood")
        backward_look_up =  Input(shape = self.edge_dim, name="bw_neighbourhood")
        return [features, foward_look_up, backward_look_up]

    def BuildNeighbourhoodLayer(self, 
                                features, 
                                look_up, 
                                hop:int, 
                                hood_func: types.LambdaType, 
                                layer_name: str, 
                                out_shape: list):

        name = layer_name if layer_name is not None else ''
        name_ext = '_lambda_init' if hop == 0 else '_lambda_step'

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
        concat = concatenate([forward_layer,backward_layer], name="fw_bw_concatenation")
        return Dense(2*hidden_dim, 
                     kernel_initializer=kernel_init,
                     activation=act,
                     kernel_regularizer=kernel_regularizer,
                     activity_regularizer=activity_regularizer,
                     name="concatenation_act")(concat)

    def GraphEmbeddingSubModelBuild(
                            self, 
                            hops: int =1,
                            aggregator: str ='mean',
                            hidden_dim: int =100, 
                            kernel_init: str ='glorot_uniform',
                            bias_init: str ='zeros',
                            act: activations = activations.relu,
                            kernel_regularizer: regularizers =regularizers.l2(0.01),
                            activity_regularizer: regularizers =regularizers.l1(0.01),
                            use_bias: bool =True,
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
        
        out_shape_lambda = (self.edge_dim[0], self.input_dim[0]+self.edge_dim[0])
        features_inputs, fw_look_up_inputs, bw_look_up_inputs = self.inputs
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

    def MakeModel(self, layers: Layer):
        """
        This function creates the model by given inputs.
            :param inputs:list=None: list of inputs
            :param layers:Layer=None: layer structure
        """
        AssertNotNone(self.inputs, 'inputs')
        AssertNotNone(layers, 'layers')
        return Model(self.inputs, layers)

    #def CompileModel(self, )

    def Summary(self, model: training.Model):
        """
        This function prints the summary of a model.
            :param model:training.Model: a build model
        """   
        AssertNotNone(model, 'plotting_tensor'), 'Plotting model was None!'
        print(model.summary())

    def Plot(self, model: training.Model, file_name: str):
        """
        This function plot and store a given model to desired file.
            :param model:Model: keras model
            :param file_name:str: name of the image file
        """ 
        AssertNotNone(model, 'plotting_tensor'), 'Plotting model was None!'
        AssertNotNone(file_name, 'name_plot_file'), 'Plot file name was None!'
        plot_model(model, to_file=file_name)

    def get_inputs(self):
        return self.inputs