import sys
import re
import os.path
import numpy as np

import copy
import random
import pickle

from nltk import word_tokenize
from collections import Counter
from sklearn.utils import shuffle
import json

import tensorflow as tf

UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
NUM = "<NUM>"

_PAD_ID = 0
_UNK_ID = 1
_SOS_ID = 2
_EOS_ID = 3


predefined_list = [PAD, UNK, SOS, EOS]
predefined_dict = {PAD : 0, UNK : 1, SOS : 2, EOS : 3}


class data_helper:

    def __init__(self, input_path, max_str_len=30, data_size=4000, embedding_size=300,  
                    vocab_dict={}, vocab_list=[], pretrained_emb=None, vocabs_given=False, pretrained_path="./glove.6B.300d.txt"):

        self.max_str_len = max_str_len
        self.data_size = data_size
        self.pretrained_path = pretrained_path
        self.input_path = input_path

        self.inputs = []

        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list

        self.batch_set = []
        
        self.pretrained_embedding = pretrained_emb
        self.vocabs_given = vocabs_given
        self.embedding_dim = embedding_size
    
    def load_inputs(self):
        if os.path.isfile(self.input_path):
            data = pickle.load(open(self.input_path, "rb"))
            self.inputs = data

        print("loading inputs done")
        return self.inputs

    def clean_string(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = string.lower()
        string = word_tokenize(string)
        string = [word for word in string]
        return string

    def clean(self):
        clean_inputs = []
        for tup in self.inputs:
            org, pp = tup
            org, pp = self.clean_string(org), self.clean_string(pp)
            clean_inputs.append((org, pp))
        self.inputs = clean_inputs
        
        return clean_inputs

    def build_vocab(self, given_vocab=[], given_vocab_dict = {}):

        vocab = Counter()

        for tup in self.inputs:
            org, pp = tup
            for word in org + pp:
                if word not in given_vocab_dict:
                    vocab[word] += 1

        vocab_common = vocab.most_common(20000)
        vocab_list = given_vocab + [x[0] for x in vocab_common]
        vocab_dict = {}
        for idx in range(len(vocab_list)):
            word = vocab_list[idx]
            vocab_dict[word] = idx
        
        print(len(vocab_dict), len(vocab_list))
        assert(len(vocab_dict) == len(vocab_list))
        return vocab_list, vocab_dict

    def build_pretrained(self, vocab, word_index):

        f = open(self.pretrained_path, 'r', encoding="utf-8", errors='ignore')
        rdr = f.readline()
        embedding_dim = len(rdr.split()) - 1
        embeddings_index = {}

        for line in f:
            line = line.strip()
            parts = line.split(' ')
            word = parts[0]
            embeddings_index[word] = np.array(parts[1:], dtype='float32')


        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        nb_words = len(word_index)
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
        return embedding_matrix 

    def id_to_token(self, string):
        tokenized = []
        for word in string:
            if word in self.vocab_dict:
                tokenized.append(self.vocab_dict[word])
            else:
                tokenized.append(self.vocab_dict[UNK])
        return tokenized

    def token_to_id(self, tokenized):
        sent = []
        for token in tokenized:
            sent.append(self.vocab_list[token])
        return sent
        
    def save_data(self):
        data_dict = {'vocab_list' : self.vocab_list,
                    'vocab_dict' : self.vocab_dict,
                    'pretrained_embedding' : self.pretrained_embedding}
        pickle.dump(data_dict, open("data_helper.p", "wb" ))

    def load_data(self):
        if os.path.isfile("data_helper.p"):
            data_dict = pickle.load(open("data_helper.p", "rb"))
            self.vocab_list = data_dict['vocab_list']
            self.vocab_dict = data_dict['vocab_dict']
            self.pretrained_embedding = data_dict['pretrained_embedding']
            self.vocabs_given = True
        else:
            print("predefined vocab file not found / creating new vocabs")

    def cut_inputs(self, max_sent_len, total_size):
        short_inputs = []
        for elem in self.inputs:
            org, pp = elem
            
            if (len(org) > 0 and len(pp) > 0):
                short_inputs.append((org[:max_sent_len], pp[:max_sent_len]))
        self.inputs = short_inputs[:total_size]
        return short_inputs

    def get_data(self, batch_size, is_test=False):

        self.batch_size = batch_size
        self.load_inputs()
        self.inputs = self.inputs[:self.data_size]
        self.clean()
        
        self.cut_inputs(self.max_str_len - 1, self.data_size)
        if (self.vocabs_given is not True):
            self.vocab_list, self.vocab_dict = self.build_vocab(given_vocab=predefined_list, given_vocab_dict=predefined_dict)
            self.pretrained_embedding = self.build_pretrained(self.vocab_list, self.vocab_dict)

        print(len(self.vocab_list))
        enc_list = []
        pp_list = []

        dec_list = []
        target_list = []
        

        def append_list(enc, pp_enc):
            enc_list.append(enc)
            pp_list.append(pp_enc)

            dec_list.append([_SOS_ID] + pp_enc)
            target_list.append(pp_enc + [_EOS_ID])


        for tup in self.inputs:
            org, pp = tup
            
            enc = self.id_to_token(org)
            pp_enc = self.id_to_token(pp)
            
            append_list(enc, pp_enc)

            #if (is_test is False):
            #    append_list(pp_enc, enc) # reverse

        print("FINAL INPUT LEN : ", len(enc_list))
            

        enc_lengths = [len(elem) for elem in enc_list]
        pp_lengths = [len(elem) for elem in pp_list]
        dec_lengths = [len(elem) for elem in dec_list] 
        
        max_enc = self.max_str_len
        max_dec = self.max_str_len

        assert (len(enc_list) == len(dec_lengths))
        assert (len(enc_list) == len(pp_list))
        assert (len(pp_list) == len(target_list))

        list_len = len(enc_list)

        for i in range(list_len):
            
            len_elem = len(enc_list[i])
            if (len_elem < max_enc):
                enc_list[i] = enc_list[i] + ([_PAD_ID] * (max_enc - len_elem))

            pp_elem = len(pp_list[i])
            if (pp_elem < max_enc):
                pp_list[i] = pp_list[i] + ([_PAD_ID] * (max_enc - pp_elem))

            dec_elem = len(dec_list[i])
            if (dec_elem < max_dec):
                dec_list[i] = dec_list[i] + ([_PAD_ID] * (max_dec - dec_elem))

            target_elem = len(target_list[i])
            if (target_elem < max_dec):
                target_list[i] = target_list[i] + ([_PAD_ID] * (max_dec - target_elem))


        
        self.data_set = list(zip(enc_list, enc_lengths, pp_list, pp_lengths, dec_list, dec_lengths, target_list))
        self.current_index = 0

        return len(self.vocab_list)

    def _get_batch_implement(self, batch_size, idx=None, is_test=False):
        if (is_test is False):

            batch = {'enc' : [],
                        'enc_len' : [],
                        'pp' : [],
                        'pp_len' : [],
                        'dec' : [],
                        'dec_len' : [],
                        'target' : []
                    }
        else:
            batch = {'enc' : [],
                        'enc_len' : [],
                        'pp' : [],
                        'pp_len' : [],
                        'target' : []
                    }
        
        if (idx is not None):
            samples = self.data_set[idx*batch_size:(idx+1)*batch_size]
        else:
            samples = random.sample(self.data_set, batch_size)

        for batch_elem in samples:
            enc_list, enc_lengths, pp_list, pp_lengths, dec_list, dec_lengths, target_list = batch_elem

            batch['enc'].append(enc_list)
            batch['enc_len'].append(enc_lengths)
            batch['pp'].append(pp_list)
            batch['pp_len'].append(pp_lengths)
            batch['target'].append(target_list)

            if (is_test is False):
                batch['dec'].append(dec_list)
                batch['dec_len'].append(dec_lengths)
                
        return batch

    def get_batch_in_index(self, batch_size, idx, is_test=False):
        return self._get_batch_implement(batch_size=batch_size, idx=idx, is_test=is_test)
        
                
    def get_next_batch(self, batch_size, is_test=False):
        return self._get_batch_implement(batch_size=batch_size, is_test=is_test)

        

def word_dropout(batch, lengths, keep_prob):

    keep_prob = np.clip(keep_prob, 0.0, 1.0)
    
    new_batch = copy.deepcopy(batch)
    new_dec_batch = []
    
    for sent_idx in range(len(new_batch)):
        for word_idx in range(lengths[sent_idx]):
            if (word_idx == 0):
                continue
            if (random.random() > keep_prob):
                new_batch[sent_idx][word_idx] = _UNK_ID
        new_dec_batch.append(new_batch[sent_idx])
        
    return new_dec_batch



class data_capsule:
    def __init__(self, batch_size, latent_size, beam_size):
        self.define_placeholders(batch_size, latent_size)
        self.beam_size = beam_size
        self.word_keep = 0.7

    def define_placeholders(self, batch_size, latent_size):
        self.source_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='source')
        self.source_lengths =  tf.placeholder(dtype=tf.int32, shape=(None,), name='source_lengths')

        self.reference_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='reference')
        self.reference_lengths =  tf.placeholder(dtype=tf.int32, shape=(None,), name='reference_lengths')

        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder')
        self.decoder_lengths =  tf.placeholder(dtype=tf.int32, shape=(None,), name='decoder_lengths')

        self.targets = tf.placeholder(dtype=tf.int32, shape=(None, None), name='targets')

        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), ())
        self.beam = tf.placeholder_with_default(tf.constant(1, dtype=tf.int32), ())

        self.latent_variable = tf.placeholder(dtype=tf.float32, shape=(None, None), name='latent_variable')


    def feed_placeholders(self, batch, is_test=False):

        if (is_test is False):
            # training
            dec = word_dropout(batch['dec'], batch['dec_len'], self.word_keep)

            feed = { self.source_inputs : batch["enc"],
                self.source_lengths : batch["enc_len"],

                self.reference_inputs : batch["pp"],
                self.reference_lengths : batch["pp_len"],

                self.decoder_inputs : dec,
                self.decoder_lengths : batch["dec_len"],

                self.targets : batch["target"],
                self.keep_prob : 0.7,
                self.beam : 1
            }
        else:
            # test
            feed = { self.source_inputs : batch["enc"],
                self.source_lengths : batch["enc_len"],

                self.reference_inputs : batch["pp"],
                self.reference_lengths : batch["pp_len"],

                self.decoder_inputs : batch["dec"],
                self.decoder_lengths : batch["dec_len"],

                self.keep_prob : 1.0,
                self.beam : self.beam_size
            }
            
        return feed
