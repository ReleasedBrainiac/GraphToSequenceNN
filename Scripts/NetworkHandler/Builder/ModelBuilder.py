from keras.models import Model
from keras.utils import plot_model
from keras import regularizers, activations
from keras.layers import Lambda, concatenate, Dense, Dropout, Input, LSTM, Embedding

from NetworkHandler.Neighbouring.NeighbourhoodCollector import Neighbourhood as Nhood
from NetworkHandler.KerasSupportMethods.SupportMethods import AssertNotNone, IsKerasTensor

class ModelBuilder():

    def CreateOneHopSubModel(neighbourhood_func, 
                         out_shape_lambda,
                         features,
                         edge_look_up,
                         hidden_dim,
                         kernel_init,
                         bias_init,
                         act,
                         kernel_regularizer,
                         activity_regularizer,
                         use_bias,
                         drop_rate):
    
        x = Lambda(neighbourhood_func, output_shape=out_shape_lambda)([features, edge_look_up])
        x = Dense(units=hidden_dim,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                activation=act,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                use_bias=use_bias)(x)
        return Dropout(drop_rate)(x)
    


def FunctionalModelBuild(input_dim,
                         edge_dim,
                         hops=1,
                         hidden_dim=100, 
                         kernel_init='glorot_uniform',
                         bias_init='zeros',
                         act=activations.relu,
                         kernel_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1(0.01),
                         use_bias=True,
                         drop_rate=0.2):  
    
    feats_dense_layer = Input(shape = input_dim, name="features")
    edge_fw_dense_layer = Input(shape = edge_dim, name="fw_neighbourhood")
    edge_bw_dense_layer =  Input(shape = edge_dim, name="bw_neighbourhood")
    
    out_shape_lambda = (edge_dim[0], input_dim[0]+edge_dim[0])
    
    neighbourhood_func = lambda x: Nhood(x[0], x[1], aggregator='mean').Execute()
    
    x = None 
    y = None 
    
    for i in range(hops):
        fw_name = ("fw_"+str(i)+"_")
        bw_name = ("bw_"+str(i)+"_")
        if i == 0:
            
            x = Lambda(neighbourhood_func, output_shape=out_shape_lambda, name=fw_name+'lambda_init')([feats_dense_layer, edge_fw_dense_layer])
            y = Lambda(neighbourhood_func, output_shape=out_shape_lambda, name=bw_name+'lambda_init')([feats_dense_layer, edge_bw_dense_layer])
        else:
            x = Lambda(neighbourhood_func, output_shape=out_shape_lambda, name=fw_name+'lambda_step')([x, edge_fw_dense_layer])
            y = Lambda(neighbourhood_func, output_shape=out_shape_lambda, name=bw_name+'lambda_step')([y, edge_bw_dense_layer])
            
        x = Dense(  units=hidden_dim,
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    activation=act,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
                    use_bias=use_bias,
                    name=fw_name+'dense_act')(x)
        
        x = Dropout(drop_rate, name=fw_name+'drop')(x)
            
        y = Dense(  units=hidden_dim,
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    activation=act,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
                    use_bias=use_bias,
                    name=bw_name+'dense_act')(y)
        
        y = Dropout(drop_rate, name=bw_name+'drop')(y)
        
    assert x is not None and y is not None
    full_concat = concatenate([x,y], name="fw_bw_concatenation")
    out = Dense(2*hidden_dim, 
                kernel_initializer=kernel_init,
                activation=act,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                name="concatenation_act")(full_concat)
    
    predictions = Dense(10, activation='softmax')(out)

    
    model = Model([feats_dense_layer, edge_fw_dense_layer, edge_bw_dense_layer], out)
    
    return model

    def Plot(model: Model, file_name: str):
        AssertNotNone(model, 'plotting_tensor'), 'Plotting model was None!'
        AssertNotNone(file_name, 'name_plot_file'), 'Plot file name was None!'
        plot_model(model, to_file=file_name)