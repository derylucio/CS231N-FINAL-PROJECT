# from datasets import dataset_utils
# import tensorflow as tf

# url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"

# # Specify where you want to download the model to
# checkpoints_dir = "../ckpts"


# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)

# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)


import numpy as np
import os
import tensorflow as tf
import urllib2
import tensorflow.contrib.slim as slim
import vgg

slim = tf.contrib.slim


class  CNN_FeatureExtractor(object):
	def __init__(self, ):
		self.endpt_str = 'vgg_16/pool5' 
		self.checkpoints_dir = "../ckpts"
		self.dummy_classes = 1000
		

	def CNNFeatureExtractor(self, input_tensor, img_height, img_width, is_training):
		print 'Starting CNNFeatureExtractor'
		with slim.arg_scope(vgg.vgg_arg_scope()):
			pool_5_net = vgg.vgg_16(input_tensor, num_classes=self.dummy_classes, is_training=is_training)
		return pool_5_net

	def getInputFn(self):
		init_fn = slim.assign_from_checkpoint_fn(
	        os.path.join(self.checkpoints_dir, 'vgg_16.ckpt'),
	        slim.get_model_variables('vgg_16'))
		return init_fn



    
    