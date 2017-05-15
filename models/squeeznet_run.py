import SqueezeNet
import tensorflow as tf

tf.reset_default_graph() # remove all existing variables in the graph 
sess = get_session() # start a new Session

# Load pretrained SqueezeNet model
SAVE_PATH = 'datasets/squeezenet.ckpt'
if not os.path.exists(SAVE_PATH):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

c_feats = sess.run(model.extract_features()[content_layer], {model.image: content_img_test})
print c_feats