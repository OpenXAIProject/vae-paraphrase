
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf
import utils


from data import data_capsule
from model import Model

config = utils.load_config("./config.json")
    
data_helper, vocab_size, test_data_helper= utils.load_data(config)

batch_size = config["batch_size"]

trainer = Model(config, vocab_size, test_data_helper.pretrained_embedding)
data_cap = data_capsule(batch_size, config["latent_dim"], config["beam_size"])
    
    
train_op, predict_op = trainer.build_train_ops(data_cap)
                                                    
   
sess = tf.Session()
sess.run(tf.global_variables_initializer())

   
test_batch = test_data_helper.get_batch_in_index(batch_size, 0)
saver = tf.train.Saver()
utils.load_model(saver, config["save_path"], sess, trainer.mode)



for n in range(config["max_iteration"]):
    
    batch = data_helper.get_next_batch(batch_size)
    feed = data_cap.feed_placeholders(batch)

    _, glob_step, loss, kl_cost, kl_rate = sess.run([train_op, trainer.global_step, trainer.loss, trainer.kl_cost, trainer.kl_rate], feed_dict=feed)

    if (glob_step % 100 == 0):
        print("::: ITER : " + str(glob_step) + "  ::: LOSS : " + str(loss) + " ::: KL : " + str(kl_cost) + " ::: KL RATE : "+ str(kl_rate))

    if (glob_step%1000 == 0):
        feed_train_test = data_cap.feed_placeholders(test_batch, is_test=True)
            
        kp = sess.run(trainer.keep_prob, feed_dict=feed)
        print("::: KEEP : " + str(kp))

        outputs = utils.random_sample_test(sess, feed_train_test, predict_op, data_cap, 3, batch_size, trainer)
        for idx in range(20):
            enc = data_helper.token_to_id(test_batch["enc"][idx])
            dec = data_helper.token_to_id(test_batch["target"][idx])
            print("org : ", " ".join(enc))
            print("pp : ", " ".join(dec))

                
            for rp in outputs:
                result = rp.predicted_ids
                unwrapped = utils.unwrap_beam(result[idx])
                neg = data_helper.token_to_id(unwrapped)
                    
                
                print("test : ", " ".join(neg))
            print("-------")


    if (glob_step!=0 and glob_step%10000 == 0):
        utils.save_model(saver, config["save_path"], sess, trainer.mode)

