import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops

import data

def build_cell(cell_hidden, keep_prob):
    cell = tf.nn.rnn_cell.LSTMCell(cell_hidden)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

def build_decoder_cell(cell_hidden, keep_prob):
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(cell_hidden, keep_prob) for n in range(2)])
    return multi_cell



class Encoder():

    def encode(self, embedding, inputs, lengths, cell_hidden, keep_prob, dtype, initial_state=(None, None), scope='Encoder', reuse=False):
        print("build_encoder/", scope)

        init_state_fw, init_state_bw =initial_state

        with tf.variable_scope(scope, reuse=reuse):

            cell_fw = build_cell(cell_hidden, keep_prob=keep_prob)
            cell_bw = build_cell(cell_hidden, keep_prob=keep_prob)

            final_input = tf.nn.embedding_lookup(embedding, inputs)
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, final_input, 
                                                                                sequence_length=lengths, 
                                                                                dtype=dtype, 
                                                                                initial_state_fw=init_state_fw,
                                                                                initial_state_bw=init_state_bw)
            return encoder_outputs, encoder_state

class Decoder():


    def __init__(self, attention_dim):
        self.decoder_cell = None
        self.att_mech = None
        self.attention_dim = attention_dim

    def build_decoder_cell(self, n_hidden, keep_prob, scope="DecoderCell", reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            self.decoder_cell = build_decoder_cell(n_hidden, keep_prob=keep_prob)
            return self.decoder_cell
    
    def build_att_mechanism(self, seq_input, seq_len, beam_width, scope='Attention', reuse=False):

        assert(self.decoder_cell != None)
        with tf.variable_scope(scope, reuse=reuse):
            tiled_seq_input = tf.contrib.seq2seq.tile_batch(seq_input, multiplier=beam_width)
            tiled_seq_len = tf.contrib.seq2seq.tile_batch(seq_len, multiplier=beam_width)

            self.att_mech = tf.contrib.seq2seq.BahdanauAttention(self.attention_dim, tiled_seq_input, memory_sequence_length=tiled_seq_len)
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, self.att_mech)

            return self.decoder_cell

    def decode(self, embedding, beam, decoder_inputs, decoder_lengths, init_state, projection_layer, latents, latent_dim, scope='Decoder', reuse=False):
        print("build_decoder/", scope)
        decoder_emb_inp = tf.nn.embedding_lookup(embedding, decoder_inputs) 

        max_len = tf.shape(decoder_inputs)[1]
        batch_size = tf.shape(decoder_inputs)[0]

        latents = tf.expand_dims(latents, axis=1)
        latents = tf.tile(latents, [1,max_len,1])
        latents = tf.reshape(latents, [batch_size, max_len, latent_dim])
        
        decoder_final_inp = tf.concat([decoder_emb_inp, latents], axis=2)
        
        
        with tf.variable_scope(scope, reuse=reuse):
            if (self.att_mech is not None):
                decoder_initial_state = self.decoder_cell.zero_state(batch_size, dtype=tf.float32)
                init_state = decoder_initial_state.clone(cell_state=init_state)

            # build training helper
            training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_final_inp, decoder_lengths)
            
            # output layer
            training_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, training_helper, init_state, projection_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder)

            # logits for the loss calculation            
            logits = outputs.rnn_output
            sample_ids = outputs.sample_id
                
            return logits, sample_ids
    

    def predict(self, embedding, beam, batch_size, init_state, projection_layer, latents, latent_dim, max_decoder_length=15, beam_width=10, scope='Decoder', reuse=False):

        print("build_predictor/", scope)
        # build predition helper
        start_tokens = tf.tile([data._SOS_ID], [batch_size])
        end_token = data._EOS_ID

        tiled_latents = tf.expand_dims(latents, axis=1)
        tiled_latents = tf.tile(tiled_latents, [1,beam,1])
        tiled_latents = tf.reshape(tiled_latents, [batch_size, beam, latent_dim]) 

        init_state = tf.contrib.seq2seq.tile_batch(init_state, multiplier=beam)

        def emb_fn(input_tokens, tiled_latents=tiled_latents, embedding_decoder=embedding):
            emb_inp = tf.nn.embedding_lookup(embedding, input_tokens)
            concat_inp = array_ops.concat([emb_inp, tiled_latents], 2)
            return concat_inp
        
        with tf.variable_scope(scope, reuse=True):

            if (self.att_mech is not None):
                decoder_initial_state = self.decoder_cell.zero_state(batch_size * beam, dtype=tf.float32)
                init_state = decoder_initial_state.clone(cell_state=init_state)

            # build predition helper
            start_tokens = tf.tile([data._SOS_ID], [batch_size])
            end_token = data._EOS_ID
           
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.decoder_cell, embedding=emb_fn, start_tokens=start_tokens, end_token=end_token,
                                                                            initial_state=init_state, beam_width=beam_width, output_layer=projection_layer)
            predict_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, maximum_iterations=max_decoder_length)
            
            return predict_outputs




class Model():

    def __init__(self, config, vocab_size, pretrained_emb):

        self.encoder = Encoder()
        self.decoder = Decoder(attention_dim=config["attention_dim"])

        self.annealing = True

        self.initializer = tf.contrib.layers.xavier_initializer()

        self.vocab_size = vocab_size
        self.embedding_size = config["emb_size"]
        self.pretrained_embedding = pretrained_emb
        self.n_hidden = config["hidden"]
        self.learning_rate = config["learning_rate"]
        self.step_size = config["step_size"]
        self.annealing_pivot = config["annealing_pivot"]
        self.latent_dim = config["latent_dim"]
        self.beam_size = config["beam_size"]

        self.reconst_weight = 100.0
        self.dtype = tf.float32

        self.embedding = self.get_embedding()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.epsilon = 0.000001

        self.mode = "SVAE"


    def define_kl_updators(self):
        g_step_float = tf.cast(self.global_step, dtype=self.dtype)
        reg = tf.constant(-self.annealing_pivot, dtype=self.dtype)
        nom = tf.constant(1000, dtype=self.dtype)
        step_term = tf.nn.tanh(tf.div(tf.add(g_step_float, reg), nom)) 

        kl_rate = tf.div(tf.add(step_term, tf.constant(1.0, dtype=self.dtype)), tf.constant(2.0, dtype=self.dtype))

        return kl_rate

    def get_embedding(self):
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable(name="embedding", shape=[self.vocab_size, self.embedding_size], 
                                                    initializer=tf.constant_initializer(np.array(self.pretrained_embedding)), trainable=False, dtype=self.dtype)
            return embedding

    def get_loss(self, label, predict, max_length, decoder_lengths, dtype):
        label_length = tf.shape(label)[1]
        logit_length = tf.shape(predict)[1]
        
        pad_size = label_length - logit_length
        predict = tf.pad(predict, [[0, 0], [0, pad_size], [0, 0]], constant_values=data._PAD_ID)
        
        max_decoder_length = max_length
        
        masks = tf.sequence_mask(lengths=decoder_lengths, 
                                    maxlen=max_decoder_length, dtype=dtype, name='masks')

        reconst_cost = tf.contrib.seq2seq.sequence_loss(logits=predict, 
                                        targets=label,
                                        weights=masks,
                                        average_across_timesteps=True,
                                        average_across_batch=True)
        return reconst_cost

    def build_encode_latent(self, output_states, batch_size, keep_prob):
        state_fw, state_bw = output_states
        init_state = [state_fw.c, state_bw.c, state_fw.h, state_bw.h]
        vector = tf.concat(axis=1, values=init_state)

        with tf.variable_scope('encode_latent'):
            vector = tf.reshape(vector, [batch_size, 4 *self.n_hidden])
            sample = tf.contrib.layers.fully_connected(vector, self.latent_dim * 2, activation_fn=tf.nn.tanh)
            sample = tf.nn.dropout(sample, keep_prob=keep_prob)

            mu = tf.contrib.layers.fully_connected(sample, self.latent_dim, activation_fn=tf.nn.tanh)
            logvar = tf.contrib.layers.fully_connected(sample, self.latent_dim, activation_fn=tf.nn.softplus)
        
            return mu, logvar

    def add_gaussian_noise(self, mu, logvar, scope='kl_sample'):
        
        with tf.variable_scope(scope):
            q_z = tf.distributions.Normal(mu, logvar)
            z = q_z.sample()
            p_z = tf.distributions.Normal(tf.zeros_like(z), tf.ones_like(z))
            kl = tf.distributions.kl_divergence(q_z, p_z, allow_nan_stats=True)
        
            kl_cost = tf.reduce_mean(tf.reduce_sum(kl, axis=-1))
            
            return z, kl_cost

    def build_train_ops(self, data_cap):

        
        self.kl_rate = kl_rate = self.define_kl_updators()

        self.keep_prob = keep_prob = data_cap.keep_prob
        batch_size = tf.shape(data_cap.source_inputs)[0]

        # encode sentences
        source_output, source_last_state = self.encoder.encode(self.embedding, data_cap.source_inputs, data_cap.source_lengths, 
                                                                        self.n_hidden, keep_prob, self.dtype, scope='SourceEncoder')
        reference_output, reference_last_state = self.encoder.encode(self.embedding, data_cap.reference_inputs, data_cap.reference_lengths, 
                                                                        self.n_hidden, keep_prob, self.dtype, initial_state=source_last_state, scope='SourceEncoder', reuse=True)

        mu, sig = self.build_encode_latent(reference_last_state, batch_size, keep_prob)

        self.not_sampled_latent = mu

        latent, kl_cost = self.add_gaussian_noise(mu, sig)
        self.sampled_latent = latent
        self.given_latent = data_cap.latent_variable

        self.kl_cost = kl_cost

        projection_layer = core.Dense(self.vocab_size, name='output_projection')
       
        # attention
        self.decoder.build_decoder_cell(self.n_hidden, keep_prob)
        source_output_concat = tf.concat([source_output[0], source_output[1]], axis=2)
        self.decoder.build_att_mechanism(source_output_concat, data_cap.source_lengths, data_cap.beam)

        # decode
        logits, self.sample_ids = self.decoder.decode(self.embedding, data_cap.beam, 
                                                            data_cap.decoder_inputs, data_cap.decoder_lengths, 
                                                            source_last_state, projection_layer, self.sampled_latent, self.latent_dim)
        self.predict_op = predict_op = self.decoder.predict(self.embedding, data_cap.beam, 
                                                            batch_size, source_last_state, projection_layer, self.given_latent, self.latent_dim, 
                                                            max_decoder_length=self.step_size, beam_width=self.beam_size, reuse=True)
        max_length = tf.shape(data_cap.decoder_inputs)[1]
        
        # loss and ops
        sequence_loss = self.get_loss(data_cap.targets, logits, max_length, data_cap.decoder_lengths, self.dtype)
        self.loss = (self.reconst_weight*sequence_loss) + (kl_rate * kl_cost)

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        self.train_op = train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        

        return train_op, predict_op
