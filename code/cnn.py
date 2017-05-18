# from datasets import dataset_utils
# import tensorflow as tf

# url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"

# # Specify where you want to download the model to
# checkpoints_dir = "../ckpts"


# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)

# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)


import tensorflow as tf
import os
#from tensorflow.contrib.slim.nets import vgg
import vgg

slim = tf.contrib.slim



class  CNN_FeatureExtractor(object):
	def __init__(self ):
		self.endpt_str = 'vgg_16/pool5' 
		self.checkpoints_dir = "../ckpts"
		self.dummy_classes = 1000
	

	def CNNFeatureExtractor(self, input_tensor, is_training):
		print 'Starting CNNFeatureExtractor'
		#with slim.arg_scope(vgg.vgg_arg_scope()):
		pool_5_net = vgg.vgg_16(input_tensor, is_training=is_training)
		print 'Extracted CNN features'
		
		return pool_5_net

	def getInputFn(self):
		
		variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
		init_fn = slim.assign_from_checkpoint_fn(
	        os.path.join(self.checkpoints_dir, 'vgg_16.ckpt'),
	        variables_to_restore) 
	        #slim.get_model_variables('vgg_16'))
		return init_fn

# inputt = tf.placeholder(tf.float32, [None,224, 224, 3])
# fs = CNN_FeatureExtractor()
# pnet = fs.CNNFeatureExtractor(inputt, True)
# init = fs.getInputFn()
# print 'done'


    
    