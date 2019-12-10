
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf
import utils

from data import data_capsule
from model import Model

config = utils.load_config("./config.json")

_, vocab_size, test_data_helper= utils.load_data(config)
tester = Model(config, vocab_size, test_data_helper.pretrained_embedding)

batch_size = config["batch_size"]

data_cap = data_capsule(batch_size, config["latent_dim"], config["beam_size"])
_, predict_op = tester.build_train_ops(data_cap)
   
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
utils.load_model(saver, config["save_path"], sess, tester.mode)

utils.write_test(test_data_helper, sess, data_cap, predict_op, 10, batch_size, tester)
