import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

def model1(input_shape, kernel_list, learn_rate, output_shape):
    """
    Build a convolutional neural network for image classification
    
    Input Args:
    input shape: Shape of the input image provided. The input shape should be 28x28x1
    kernel_list: List of kernel level information. Its a list of list containing per 
                    level number of kernels and the kernel shape of the model.
    learn_rate: Learning rate of the optimizer
    output_shape: Shape of the final layer of the model. 10 in our case
    
    Output Args:
    model: Tensorflow model which is compiled
    """
    def conv_block(num_kernels, kernel_shape, stride_shape,input_layer, layer_num):
        """
        Create a convolution block which is a convolution layer followed by activation
        
        Input Args:
        num_kernels: Number of kernels in the convolution layer
        kernel_shape: Shape of the kernels in the convolution layer
        stride_shape: Shape of the strides in the convolution layer
        input_layer: Input layer connected to the block
        layer_num: Layer number in the model
        
        Output Args:
        out: Output Layer of the convolution block
        """
        conv = layers.Conv2D(num_kernels, 
                                 kernel_shape, 
                                 padding='valid', 
                                 activation='relu',
                                 strides=stride_shape,
                                 name = 'conv'+str(layer_num))(input_layer)
        bn = layers.BatchNormalization(name = 'batchnorm'+str(layer_num))(conv)
        out = layers.MaxPool2D(pool_size=(3, 3), 
                             strides=(1, 1), 
                             name='maxpool'+str(layer_num),
                             padding='same')(bn)
        return out
    
    # Create convolution block
    input_layer = layers.Input(shape=input_shape, name='input_layer')
    # Connect input image to the first convolution block
    x = conv_block(kernel_list[0][0], kernel_list[0][1], kernel_list[0][2],input_layer, 1)
    # Connect rest of the block to each other
    for i in range(1, len(kernel_list)):
        kernel = kernel_list[i]
        x = conv_block(kernel[0], kernel[1], kernel[2],x, i+1)
    
    # Connect Flatten layers and add output layer
    flatten = layers.Flatten(name='flatten')(x)
    dense1 = layers.Dense(50, name = 'dense1')(flatten)
    out_layer = layers.Dense(output_shape, activation='softmax', name = 'dense2')(dense1)
    
    # Define input and outputs of the model
    model = models.Model(input_layer, out_layer)
    
    # Compile the model
    adam = optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model