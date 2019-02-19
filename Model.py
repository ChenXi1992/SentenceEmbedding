from keras.layers import Dense, Bidirectional,Input, Dropout, Flatten, concatenate, dot, GaussianDropout, Activation, GRU
from keras.models import Model
from keras.regularizers import l1, l2

import re

def buildModel(input_X,kernel_reg,num_neurons,merge_mode):
    org_input = Input(shape=(input_X.shape[1],input_X.shape[2]), name = 'org_input')
    org_input_1 = Bidirectional(GRU(num_neurons[0],return_sequences= True,kernel_regularizer=l2(kernel_reg)), merge_mode= merge_mode,name='X_input_first')(org_input)
    org_input_2 = Bidirectional(GRU(num_neurons[1],kernel_regularizer=l2(kernel_reg)), merge_mode= merge_mode,name='X_input_second')(org_input_1)

    # Create shared layer for masked sentence 
    shared_first = Bidirectional(GRU(num_neurons[0],return_sequences= True,kernel_regularizer=l2(kernel_reg)), merge_mode= merge_mode,name='X_input_mask_1')
    shared_second = Bidirectional(GRU(num_neurons[1],kernel_regularizer=l2(kernel_reg)), merge_mode= merge_mode,name='X_input_mask_2')

    mask_input_first = Input(shape=(input_X.shape[1],input_X.shape[2]), name = 'mask_input_first')
    mask_input_first_1 = shared_first(mask_input_first)
    mask_input_first_2 = shared_second(mask_input_first_1)

    mask_input_second = Input(shape=(input_X.shape[1],input_X.shape[2]), name = 'mask_input_second')
    mask_input_second_1 = shared_first(mask_input_second)
    mask_input_second_2 = shared_second(mask_input_second_1)

    mask_input_third = Input(shape=(input_X.shape[1],input_X.shape[2]), name = 'mask_input_third')
    mask_input_third_1 = shared_first(mask_input_third)
    mask_input_third_2 = shared_second(mask_input_third_1)

    out1 = dot([org_input_2, mask_input_first_2], axes=1, name='output_1')
    out2 = dot([org_input_2, mask_input_second_2], axes=1, name='output_2')
    out3 = dot([org_input_2, mask_input_third_2], axes=1, name='output_3')

    concat = concatenate([out1, out2, out3], name='concat')

    output = Activation('softmax')(concat)

    model = Model(inputs=[org_input,mask_input_first,mask_input_second,mask_input_third], outputs=output)

    return model

def extractHiddenState(layerName,model,predict_input):
    
    outputs = []

    for i in range(len(layerName)):
        outputs.append(model.get_layer(layerName[i]).get_output_at(0))

    input_layer = Model(inputs=model.input,
                        outputs=outputs)

    input_layer_embed = input_layer.predict([predict_input,predict_input,predict_input,predict_input])
    
    return input_layer_embed


