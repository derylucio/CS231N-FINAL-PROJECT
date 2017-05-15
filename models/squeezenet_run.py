import squeezenet as sn
import tensorflow as tf
import numpy as np
import os

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    session = tf.Session(config=config)
    return session

tf.reset_default_graph() # remove all existing variables in the graph 
sess = tf.Session() # get_session() # start a new Session

# Load pretrained SqueezeNet model
SAVE_PATH = '../datasets/squeezenet.ckpt'
if not os.path.exists(SAVE_PATH):
	pass
    # raise ValueError("You need to download SqueezeNet!")
model = sn.SqueezeNet(save_path=SAVE_PATH, sess=sess)

# Not yet sure how to test this. 
img = np.random.randn(1, 32, 32, 3)
content_layer = -1
feats = sess.run(model.extract_features()[content_layer], {model.image: img})
print feats