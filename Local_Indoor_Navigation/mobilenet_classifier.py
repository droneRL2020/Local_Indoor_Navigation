'''
This is a re-written study sciprt of the original code below,
https://github.com/Zehaos/MobileNet/blob/master/nets/mobilenet.py
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim

def mobilenet(inputs,
              num_classes=10,
              is_training=True,
              width_multiplier=1,
              scope='mobilenet_classifier'):
    """ MobileNet_classifier_pracitce
    Args:
        inputs: a tensor size [batch_size, height, width, channels]
        num_classes: number of predicted classes(Cat or Dog)
        is_training: whether or not the model is being trained
        width_multiplier
        scope:Optional scope for the variables
    Returns:
        logits: the pre-softmax activations, a tensor of size [batch_size, 'num_classes']
        end_points: a dictionary from components of the network to the corresponding activation (?)
    """

    def _depthwise_separable_conv(inputs, 
                                 num_pwc_filters, 
                                 width_multiplier, 
                                 sc, 
                                 downsample=False):
        """ Helper function to build the depth-wise seprable convolution layer.
        """
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1
        
        # skip depthwise by setting num_outpus=None
        depthwise_conv = slim.separable_conv2d(inputs,
                                                num_outputs=None,
                                                stride=_stride,
                                                depth_multiplier=1, 
                                                kernel_size=[3, 3],
                                                scope=sc+'/depthwise_conv')
        
        bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
        pointwise_conv = slim.conv2d(bn,
                                     num_pwc_filters,
                                     kernel_size=[1,1],
                                     scope=sc+'/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return bn
    
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None, 
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm], 
                                is_training=is_training, 
                                activation_fn=tf.nn.relu,
                                fused=True):
                net = slim.conv2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
                net = slim.batch_norm(net, scope='conv_1/batch_norm')
                net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')
                net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
                net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')
                
        
        
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        end_points['squeeze'] = net
        hypothesis = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_16')
        predictions = slim.softmax(hypothesis, scope='Predictions')
        
        end_points['Logits'] = hypothesis
        end_points['Predictions'] = predictions            
        
    return hypothesis, end_points



mobilenet.default_image_size = 224



def mobilenet_arg_scope(weight_decay=0.0):
    '''Defines the default mobilenet argument scope.
    
    Args:
        weight_decay: The weight decay to use for regularizing th emodel.
        
    Returns:
        An 'arg_scope' to use for the MobileNet model.
    '''
    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_initializer=slim.initializers.xavier_initializer(),
        biases_initializer=slim.init_ops.zeros_initializer(),
        weight_regularizzer=slim.l2_regularizer(weight_decay)) as sc:
        
        return sc
    
                             
                             