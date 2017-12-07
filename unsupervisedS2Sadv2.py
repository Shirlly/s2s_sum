# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:59:15 2017

@author: Zheng Xin
"""

"""
This file is the algorithm for the proposed unsupervised sequence to sequence model.
The model is used for Abstractive Summarization Generation.
"""

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, GRUCell, LSTMStateTuple, DropoutWrapper, MultiRNNCell


class unsuS2S_adv(object):
    def __init__(self, vocab_size, embed_dim, batch_size, pre_embed, args, attention=False):
        self.PAD = 0
        self.GO = 1
        self.EOS = 2

        with tf.Graph().as_default() as self.graph:
            self.sess = tf.Session()
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.attention = attention
            self.alpha = args.alpha
            self.learning_rate = args.learning_rate
            self.encoder_dropout = args.encoder_dropout
            self.decoder_dropout = args.decoder_dropout
            self.multi_layer = args.multi_layer
            self.hidden = args.all

            self.encoder_hidden_nodes = 128
            self.decoder_hidden_nodes = self.encoder_hidden_nodes * 2

            self.encoder_input = tf.placeholder('int32', [None, None], name='encoder_input')  # each input is a tensor
            self.en_in_len = tf.placeholder('int32', [None, ], name='encoder_input_length')
            self.de_out = tf.placeholder('int32', [None, None], name='decoder_output')
            self.de_out_len = tf.placeholder('int32', [None, ], name='decoder_output_length')
            self.decoder_mask = tf.placeholder(bool, [None, None], name='decoder_loss_mask')

            self.lead = tf.placeholder('int32', [None, None], name='decoder_lead')
            self.lead_out_len = tf.placeholder('int32', [None, ], name='decoder_lead_length')

            self._init_decoder_train_connectors()

            self.pre_embed = pre_embed
            self.embed = tf.Variable(tf.random_uniform([self.vocab_size+3, self.embed_dim], -1.0, 1.0), 'float32', name='embedding')
            self._preEmbed_replace()
            self.encoder_input_embed = tf.nn.embedding_lookup(self.embed, self.encoder_input)
            self.decoder_train_input_embed = tf.nn.embedding_lookup(self.embed, self.decoder_train_input)

            self.W = tf.Variable(tf.random_uniform([self.decoder_hidden_nodes, self.vocab_size], -1, 1), 'tf.float32')
            self.b = tf.Variable(tf.zeros([self.vocab_size]), 'tf.float32')
            self.global_step = tf.Variable(0, name='global_step', trainable=False)


    def _preEmbed_replace(self):
        for ele in self.pre_embed:
            self.embed[ele].assign(self.pre_embed[ele])

    def _init_decoder_train_connectors(self):
        scope = 'FeedDecoderTrain'
        with self.graph.as_default():
            with tf.name_scope(scope):
                sequence_size, self.batch_size = tf.unstack(tf.shape(self.de_out))

                GO_SLICE = tf.ones([1, self.batch_size], dtype=tf.int32) * self.GO
                PAD_SLICE = tf.ones([1, self.batch_size], dtype=tf.int32) * self.PAD

                self.decoder_train_input = tf.concat([GO_SLICE, self.de_out], axis=0)
                self.decoder_train_length = self.de_out_len + 1

                decoder_train_targets = tf.concat([self.de_out, PAD_SLICE], axis=0)
                decoder_lead = tf.concat([self.lead, PAD_SLICE], axis=0)

                decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
                decoder_lead_seq_len, _ = tf.unstack(tf.shape(decoder_lead))

                decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                            decoder_train_targets_seq_len,
                                                            on_value=self.EOS, off_value=self.PAD,
                                                            dtype=tf.int32)

                decoder_lead_eos_mask = tf.one_hot(self.lead_out_len,
                                                   decoder_lead_seq_len,
                                                   on_value=self.EOS, off_value=self.PAD,
                                                   dtype=tf.int32)

                decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
                decoder_lead_eos_mask = tf.transpose(decoder_lead_eos_mask, [1, 0])

                decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)
                decoder_lead = tf.add(decoder_lead, decoder_lead_eos_mask)

                self.decoder_train_targets = decoder_train_targets
                self.decoder_lead = decoder_lead

                self.loss_weights = tf.ones([self.batch_size, tf.reduce_max(self.decoder_train_length)],
                                            dtype=tf.float32, name='loss_weights')

    def encoder(self):
        scope = 'Encoder'
        with self.graph.as_default():
            with tf.name_scope(scope):
                encoder_cell = GRUCell(self.encoder_hidden_nodes)  # LSTM
                if self.encoder_dropout == True:
                    encoder_cell = DropoutWrapper(cell=encoder_cell, output_keep_prob=0.5)
                if self.multi_layer == True:
                    encoder_cell = MultiRNNCell([encoder_cell] * 2)

                # -- Bidirectional LSTM --
                encoder_input_embed = tf.nn.embedding_lookup(self.embed, self.encoder_input)
                len = self.en_in_len
                ((encoder_fw_output, encoder_bw_output),
                 (encoder_fw_final_state, encoder_bw_final_state)) \
                    = (tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = encoder_cell, cell_bw = encoder_cell,
                    inputs = encoder_input_embed, time_major = True,
                    sequence_length=len, dtype=tf.float32))

                if self.multi_layer == True:
                    self.encoder_output = tf.concat((encoder_fw_output, encoder_bw_output), 2)
                    self.encoder_final_state = tf.concat((encoder_fw_final_state[1], encoder_bw_final_state[1]), 1)
                else:
                    self.encoder_output = tf.concat((encoder_fw_output, encoder_bw_output), 2)
                    # GRU encoder output state
                    self.encoder_final_state = tf.concat((encoder_fw_final_state, encoder_bw_final_state), 1)
                # LSTM encoder output state
                # encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
                # encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
                # self.encoder_final_state = LSTMStateTuple(c = encoder_final_state_c, h = encoder_final_state_h)

        return self.encoder_input, self.en_in_len, self.encoder_output, self.encoder_final_state


    def loop_fn_initial(self):
        # Loop initial state for decoder, input: encoder_final_state and embeddings
        initial_elements_finished = (0 >= self.de_out_len)  # all False at initial step
        eos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='EOS')
        eos_step_embed = tf.nn.embedding_lookup(self.embed, eos_time_slice)
        initial_input = eos_step_embed
        initial_cell_state = self.encoder_final_state
        initial_cell_output = None
        initial_loop_state = None
        return (initial_elements_finished, initial_input, initial_cell_state,
                initial_cell_output, initial_loop_state)


    def loop_fn_transition(self, time, previous_output, previous_state,previous_loop_state):
        # -- Transition function: pass previous generated token to current state --

        def get_next_input():
            output_logits = tf.add(tf.matmul(previous_output, self.W), self.b)
            prediction = tf.argmax(output_logits, axis=1)
            next_input = tf.nn.embedding_lookup(self.embed, prediction)
            return next_input

        elements_finished = (time >= self.de_out_len)

        finished = tf.reduce_all(elements_finished)
        pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')
        pad_step_embed = tf.nn.embedding_lookup(self.embed, pad_time_slice)
        input = tf.cond(finished, lambda: pad_step_embed, get_next_input)
        state = previous_state
        output = previous_output
        loop_state = None
        return (elements_finished, input, state, output, loop_state)

    def loop_fn(self, time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:
            assert previous_output is None and previous_loop_state is None
            return self.loop_fn_initial()
        else:
            return self.loop_fn_transition(time, previous_output,
                                           previous_state, previous_loop_state )

    def decoder(self, max_twee_len):
        scope = 'Decoder'
        with self.graph.as_default():
            with tf.name_scope(scope):
                decoder_cell = GRUCell(self.decoder_hidden_nodes)

                encoder_max_time, self.batch_size = tf.unstack(tf.shape(self.encoder_input))

                # self.decoder_length = self.en_in_len + 3
                self.decoder_length = max_twee_len + 3

                # -- Simple RNN --
                # decoder_output, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                #                                                         decoder_input_embed,
                #                                                         'tf.float32',
                #                                                         initial_state = self.encoder_final_state,
                #                                                         time_major = True,
                #                                                         scope="plain_decoder")
                # decoder_logits = tf.contrib.layers.linear(decoder_output, self.vocab_size)
                # decoder_prediction = tf.argmax(decoder_logits, 2)
                assert self.EOS == 1 and self.PAD == 0
                # -- Complex mechanism for decoder: with previous generated tokens, or with attention
                # import pdb; pdb.set_trace()
                decoder_output_ta, decoder_final_state, _ = \
                    tf.nn.raw_rnn(decoder_cell, self.loop_fn)

                decoder_output = decoder_output_ta.stack()

                decoder_max_step, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_output))
                decoder_output_flat = tf.reshape(decoder_output, (-1, decoder_dim))
                decoder_logits_flat = tf.add(tf.matmul(decoder_output_flat, self.W), self.b)
                decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_step, decoder_batch_size, self.vocab_size))
                decoder_prediction = tf.argmax(decoder_logits, 2)

        return self.de_out, self.de_out_len, decoder_logits, decoder_prediction

    def decoder_adv(self, max_twee_len):
        with self.graph.as_default():
            with tf.variable_scope("Decoder") as scope:
                # self.decoder_length = max_twee_len + 4

                self.decoder_length = tf.ones([self.batch_size, ], dtype='int32') * (max_twee_len + 4)

                def output_fn(outputs):
                    return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

                self.decoder_cell = GRUCell(self.decoder_hidden_nodes)
                if self.decoder_dropout == True:
                    self.decoder_cell = DropoutWrapper(cell=self.decoder_cell, output_keep_prob=0.5)
                if not self.attention:
                    decoder_train = seq2seq.simple_decoder_fn_train(encoder_state = self.encoder_final_state)
                    decoder_inference = seq2seq.simple_decoder_fn_inference(
                        output_fn=output_fn,
                        encoder_state=self.encoder_final_state,
                        embeddings=self.embed,
                        start_of_sequence_id = self.EOS,
                        end_of_sequence_id = self.EOS,
                        maximum_length = self.decoder_length,
                        num_decoder_symbols = self.vocab_size
                    )
                else:
                    # attention_states: size [batch_size, max_time, num_units]
                    self.attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
                    (self.attention_keys, self.attention_values, self.attention_score_fn, self.attention_construct_fn) = \
                        seq2seq.prepare_attention(attention_states=self.attention_states, attention_option="bahdanau",
                                                  num_units=self.decoder_hidden_nodes)

                    decoder_fn_train = seq2seq.attention_decoder_fn_train(
                        encoder_state=self.encoder_final_state, attention_keys=self.attention_keys,
                        attention_values=self.attention_values, attention_score_fn=self.attention_score_fn,
                        attention_construct_fn=self.attention_construct_fn, name="attention_decoder"
                    )

                    self.decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                        output_fn = output_fn,
                        encoder_state = self.encoder_final_state,
                        attention_keys = self.attention_keys,
                        attention_values = self.attention_values,
                        attention_score_fn = self.attention_score_fn,
                        attention_construct_fn = self.attention_construct_fn,
                        embeddings = self.embed,
                        start_of_sequence_id = self.EOS,
                        end_of_sequence_id = self.EOS,
                        maximum_length = 23, # max_twee_len + 3,  #tf.reduce_max(self.de_out_len) + 3,
                        num_decoder_symbols = self.vocab_size
                    )
                    self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                        self.embed, self.decoder_train_input)
                    (self.decoder_outputs_train, self.decoder_state_train, self.decoder_context_state_train) = (
                        seq2seq.dynamic_rnn_decoder(cell = self.decoder_cell, decoder_fn = decoder_fn_train,
                                                    inputs = self.decoder_train_inputs_embedded,
                                                    sequence_length=self.decoder_length,  # self.decoder_train_length
                                                    time_major = True, scope = scope)
                    )

                    self.decoder_logits_train = output_fn(self.decoder_outputs_train)
                    self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                              name='decoder_prediction_train')

                    scope.reuse_variables()
                    (self.decoder_logits_inference, self.decoder_state_inference, self.decoder_context_state_inference) = (
                        seq2seq.dynamic_rnn_decoder(
                            cell=self.decoder_cell, decoder_fn=self.decoder_fn_inference, time_major=True, scope=scope
                        )
                    )
                    self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1,
                                                                  name='decoder_prediction_inference')

        return self.de_out, self.de_out_len, self.decoder_logits_train, self.decoder_prediction_train, \
               self.loss_weights, self.decoder_train_targets, self.decoder_prediction_inference, \
               self.lead, self.lead_out_len, self.decoder_lead, self.decoder_outputs_train, self.decoder_state_train


    def multi_cost(self, args):
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_train_targets, depth=self.vocab_size, dtype='float32'),
            logits=self.decoder_logits_train
        )
        """
        bias_loss = tf.losses.hinge_loss(
            labels=tf.one_hot(self.decoder_lead, depth=self.vocab_size, dtype='float32'),
            logits=self.decoder_logits_train
        )
        bias_loss = tf.losses.hinge_loss(
            labels=self.encoder_final_state,
            logits=self.decoder_state_train
        )"""
        if self.hidden == 'avg':
            encoder_hidden = tf.reduce_mean(self.encoder_output, 0)
            decoder_hidden = tf.reduce_mean(self.decoder_outputs_train, 0)
        elif self.hidden == 'sum':
            encoder_hidden = tf.reduce_sum(self.encoder_output, 0)
            decoder_hidden = tf.reduce_sum(self.decoder_outputs_train, 0)

        bias_loss = tf.nn.l2_loss(encoder_hidden - decoder_hidden)

        # loss_ = tf.reduce_mean(tf.boolean_mask(stepwise_cross_entropy, self.decoder_mask))
        loss_ = tf.reduce_mean(stepwise_cross_entropy)
        bias_ = tf.reduce_mean(bias_loss)
        tlt_loss = loss_ + float(self.alpha) * bias_

        learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step,
                                                   5000, 0.96, staircase=True)

        if args.opt == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=args.adagrad_init_acc)
        elif args.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif args.opt == 'RMS':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)

        if args.opt_clip == True:
            tvars = tf.trainable_variables()
            gradients = tf.gradients(tlt_loss, tvars)
            grads, global_norm = tf.clip_by_global_norm(gradients, 1)
            tf.summary.scalar('global_norm', global_norm)
            _train_op = optimizer.apply_gradients(zip(grads, tvars), name='train_step')
        else:
            _train_op = optimizer.minimize(tlt_loss)
        # train_ = tf.train.AdamOptimizer(self.learning_rate).minimize(tlt_loss)  # loss_
        return tlt_loss, _train_op, self.decoder_mask


    def decode_topk(self, max_twee_len, input_state, pos, beam_size):
        with self.graph.as_default():
            with tf.variable_scope("Decoder") as scope:
                tf.get_variable_scope().reuse_variables()
                def output_fn(outputs):
                    return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=input_state,  # self.encoder_final_state
                    attention_keys=self.attention_keys,
                    attention_values=self.attention_values,
                    attention_score_fn=self.attention_score_fn,
                    attention_construct_fn=self.attention_construct_fn,
                    embeddings=self.embed,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length= 23, # max_twee_len + 3,
                    num_decoder_symbols=self.vocab_size
                )
                (self.decoder_logits_inference, self.decoder_state_inference, self.decoder_context_state_inference) = (
                    seq2seq.dynamic_rnn_decoder(
                        cell=self.decoder_cell, decoder_fn=decoder_fn_inference, time_major=True, scope=scope
                    )
                )
                self._topk_log_probs, self._topk_ids = tf.nn.top_k(
                    tf.log(tf.nn.softmax(self.decoder_logits_inference[pos])), beam_size)

        return self._topk_log_probs, self._topk_ids, self.decoder_state_inference


    def TweetInitDecoder(self, input_state):
        with self.graph.as_default():
            with tf.variable_scope("TweetInitDecoder") as scope:
                def output_fn(outputs):
                    return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=input_state,  # self.encoder_final_state
                    attention_keys=self.attention_keys,
                    attention_values=self.attention_values,
                    attention_score_fn=self.attention_score_fn,
                    attention_construct_fn=self.attention_construct_fn,
                    embeddings=self.embed,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=23,  # max_twee_len + 3,
                    num_decoder_symbols=self.vocab_size
                )
                (self.tidecoder_logits_inference, self.tidecoder_state_inference, self.tidecoder_context_state_inference) = (
                    seq2seq.dynamic_rnn_decoder(
                        cell=self.decoder_cell, decoder_fn=decoder_fn_inference, time_major=True, scope=scope
                    )
                )

                self.tidecoder_prediction_inference = tf.argmax(self.tidecoder_logits_inference, axis=-1,
                                                                name='TIdecoder_prediction_inference')
        return self.tidecoder_prediction_inference
