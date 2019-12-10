from __future__ import absolute_import, division

import os
import sys
import pickle

import tensorflow as tf
import numpy as np

from data import data_helper
import json

def load_config(file_path):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
        return config

def load_data(config):

    max_str_len = config["step_size"]
    max_data_size =config["max_data_size"] 
    emb_size = config["emb_size"]
    pretrained_emb_path = config["pretrained_embedding_path"]
    batch_size = config["batch_size"]

    train_helper = data_helper(input_path=config["train_path"], max_str_len=max_str_len, data_size=max_data_size, embedding_size=emb_size, pretrained_path=pretrained_emb_path)
    train_helper.load_data()
    _ = train_helper.get_data(batch_size)


    test_helper = data_helper(max_str_len=max_str_len, input_path=config["test_path"], data_size=max_data_size, embedding_size=emb_size,
                                            vocab_dict=train_helper.vocab_dict, vocab_list=train_helper.vocab_list, pretrained_emb=train_helper.pretrained_embedding, 
                                            vocabs_given=True)

    vocab_size = test_helper.get_data(batch_size, is_test=True)

    test_helper.save_data()

    return train_helper, vocab_size, test_helper

def unwrap_beam(sent_elem):
    new_elem = []
    for step_idx in range(len(sent_elem)):
        new_step = sent_elem[step_idx][0]
        new_elem.append(new_step)
    return new_elem

def write_sent_to_file(p_file, sent, helper):
    sent = helper.token_to_id(sent)
    sent = " ".join(sent)
    sent = sent.split("<EOS>")[0]
    if (len(sent) < 0):
        sent = " "
    p_file.write(sent)
    p_file.write("\n")

def write_test(d_helper, sess, data_cap, op, num, batch_size, model, path="./"):
    ref_f_name = "/reference"
    out_f_name = "/hypothesis"

    reference_file = open(path + ref_f_name + '.txt', 'w')
    output_files = []
    total_sample = num

    for out_iter in range(total_sample):
        of = open(path + out_f_name + '_' + str(out_iter) + '.txt', 'w')
        output_files.append(of)

    total_len = len(d_helper.data_set)

    for batch_idx in range(total_len//batch_size):

        batch = d_helper.get_batch_in_index(batch_size, batch_idx)
        feed = data_cap.feed_placeholders(batch, is_test=True)
        p_list = random_sample_test(sess, feed, op, data_cap, num, batch_size, model)
        assert(len(p_list) == total_sample)
        
        batch_len = len(batch['enc'])
        for idx in range(batch_len):
            write_sent_to_file(reference_file, batch['pp'][idx], d_helper)

        for out_iter in range(total_sample):
            for idx in range(batch_len):
                generated_sample = p_list[out_iter].predicted_ids[idx]
                generated_sample = unwrap_beam(generated_sample)
                write_sent_to_file(output_files[out_iter], generated_sample , d_helper)

            
    reference_file.close() 
    for output_file in output_files:
        output_file.close()


def random_sample_test(sess, feed, op, data_cap, num, batch_size, model):
    
    output_list = []
    for _ in range(num):
        latents = sess.run(model.sampled_latent, feed_dict=feed)
        assert(len(latents)==batch_size)
        latents = np.array(latents)
        
        feed[data_cap.latent_variable] = latents

        result = sess.run(op, feed_dict=feed)
        output_list.append(result)

    return output_list


def load_model(saver, load_path, session, mode):
    model_path = load_path + "/"+ mode
    if (os.path.exists(model_path + ".meta")):
        saver.restore(session, model_path)
        print("::   model loading")
    else:
        print("::   model dose not exist")
        session.run(tf.global_variables_initializer())
    return None


def save_model(saver, save_path, session, mode):
    saver.save(session, save_path + "/" + mode)
    print("::   model saving")
    return None
