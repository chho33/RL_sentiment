import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
#from tensorflow.contrib.rnn.python.ops import core_rnn
#from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from six.moves import range
import numpy as np
import random
import copy

import data_utils
import seq2seq

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

class Seq2seq():
  
  def __init__(self,
               vocab_size,
               buckets,
               size,
               num_layers,
               batch_size,
               mode,
               input_keep_prob,
               output_keep_prob,
               state_keep_prob,
               beam_search,
               beam_size,
               schedule_sampling='linear', 
               sampling_decay_rate=0.99,
               sampling_global_step=150000,
               sampling_decay_steps=500
               ):
    
    self.vocab_size = vocab_size
    self.buckets = buckets
    # units of rnn cell
    self.size = size
    # dimension of words
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(0.5, trainable=False)
    self.mode = mode
    self.dummy_reply = ["what ?", "yeah .", "you are welcome ! ! ! !"]

    # learning rate decay
    self.learning_rate_decay = self.learning_rate.assign(self.learning_rate * 0.99) 

    # input for Reinforcement part
    self.loop_or_not = tf.placeholder(tf.bool)
    self.reward = tf.placeholder(tf.float32, [None])
    batch_reward = tf.stop_gradient(self.reward)
    self.RL_index = [None for _ in self.buckets]

    # dropout
    self.input_keep_prob =  input_keep_prob
    self.output_keep_prob = output_keep_prob
    self.state_keep_prob =  state_keep_prob

    # beam search
    self.beam_search = beam_search
    self.beam_size = beam_size

    # schedule sampling
    self.schedule_sampling = schedule_sampling
    if self.schedule_sampling == 'False': self.schedule_sampling = False
    self.sampling_probability = 1.0
    self.sampling_global_step = sampling_global_step
    self.sampling_decay_steps = sampling_decay_steps 
    self.sampling_decay_rate = sampling_decay_rate 

    if self.schedule_sampling == 'linear':
      self.decay_fixed = self.sampling_probability * (self.sampling_decay_steps / self.sampling_global_step)
      self.sampling_probability = tf.Variable(self.sampling_probability, trainable=False)
      self.sampling_probability_decay = tf.assign_sub(self.sampling_probability, self.decay_fixed)
      self.sampling_probability = tf.maximum(self.sampling_probability,tf.constant(0.0))
    elif self.schedule_sampling == 'exp':
      self.sampling_probability = tf.Variable(self.sampling_probability, trainable=False)
      #self.sampling_probability = tf.train.exponential_decay(
      self.sampling_probability_decay = tf.assign(
        self.sampling_probability,
        tf.train.natural_exp_decay(
          self.sampling_probability,
          self.sampling_global_step,
          self.sampling_decay_steps,
          self.sampling_decay_rate,
          staircase = True)
      )
    elif self.schedule_sampling == 'inverse_sigmoid':
      self.sampling_probability = tf.Variable(self.sampling_probability, trainable=False)
      self.sampling_probability_decay = tf.assign(
        self.sampling_probability,
        #tf.train.cosine_decay(
        tf.train.linear_cosine_decay(
          self.sampling_probability,
          self.sampling_decay_steps,
          self.sampling_global_step,
        )
      )
    elif not self.schedule_sampling:
      pass
    else:
      raise ValueError("schedule_sampling must be one of the following: [linear|exp|inverse_sigmoid|False]")

    w_t = tf.get_variable('proj_w', [self.vocab_size, self.size])
    w = tf.transpose(w_t)
    b = tf.get_variable('proj_b', [self.vocab_size])
    output_projection = (w, b)

    def sample_loss(labels, inputs):
      labels = tf.reshape(labels, [-1, 1])
      local_w_t = tf.cast(w_t, tf.float32)
      local_b = tf.cast(b, tf.float32)
      local_inputs = tf.cast(inputs, tf.float32)
      return tf.cast(tf.nn.sampled_softmax_loss(weights = local_w_t,
                                                biases = local_b,
                                                inputs = local_inputs,
                                                labels = labels,
                                                num_sampled = 512,
                                                num_classes = self.vocab_size),
                                                dtype = tf.float32)
    softmax_loss_function = sample_loss

    #FIXME add RL function
    def seq2seq_multi(encoder_inputs, decoder_inputs, mode):
      embedding = tf.get_variable("embedding", [self.vocab_size, self.size])
      loop_function_RL = None
      if mode == 'MLE':
        feed_previous = False
      elif mode == 'TEST':
        feed_previous = True
      # need loop_function
      elif mode == 'RL':
        feed_previous = True

        def loop_function_RL(prev, i):
          prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
          prev_index = tf.multinomial(tf.log(tf.nn.softmax(prev)), 1)
          
          if i == 1:
            for index, RL in enumerate(self.RL_index):
              if RL is None:
                self.RL_index[index] = prev_index
                self.index = index
                break
          else:
            self.RL_index[self.index] = tf.concat([self.RL_index[self.index], prev_index], axis = 1)
          #self.RL_index: [(?,9),(?,14),(?,24),(?,49)]
          #RL_index指的是取樣後每個字的index
          prev_index = tf.reshape(prev_index, [-1])
          #prev_index: (?,)
          # decide which to be the next time step input
          sample = tf.nn.embedding_lookup(embedding, prev_index)
          #sample: (?,256)
          from_decoder = tf.nn.embedding_lookup(embedding, decoder_inputs[i])
          #from_decoder: (?,256)
          return tf.where(self.loop_or_not, sample, from_decoder)
      self.loop_function_RL = loop_function_RL

      return seq2seq.embedding_attention_seq2seq(
             encoder_inputs,
             decoder_inputs,
             cell,
             num_encoder_symbols = self.vocab_size,
             num_decoder_symbols = self.vocab_size,
             embedding_size = self.size,
             output_projection = output_projection,
             feed_previous = feed_previous,
             dtype = tf.float32,
             embedding = embedding,
             beam_search = self.beam_search,
             beam_size = self.beam_size,
             loop = loop_function_RL,
             schedule_sampling = self.schedule_sampling,
             sampling_probability = self.sampling_probability)
    
    # inputs
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []

    for i in range(buckets[-1][0]):
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape = [None],
                                                name = 'encoder{0}'.format(i)))
    for i in range(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape = [None],
                                                name = 'decoder{0}'.format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape = [None],
                                                name = 'weight{0}'.format(i)))
    targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

    def single_cell():
      return tf.contrib.rnn.GRUCell(self.size)
      #return tf.contrib.rnn.BasicLSTMCell(self.size)
    cell = single_cell()
    if self.num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])
      cell = rnn.DropoutWrapper(cell,input_keep_prob=self.input_keep_prob,output_keep_prob=self.output_keep_prob,state_keep_prob=self.state_keep_prob)

    if self.mode == 'MLE':
      self.outputs, self.losses = seq2seq.model_with_buckets(
           self.encoder_inputs, self.decoder_inputs, targets,
           self.target_weights, self.buckets, lambda x, y: seq2seq_multi(x, y, self.mode),
           softmax_loss_function = softmax_loss_function)
      
      for b in range(len(self.buckets)):
        self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1]
                           for output in self.outputs[b]]

      self.update = []
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in range(len(self.buckets)):
        gradients = tf.gradients(self.losses[b], tf.trainable_variables()) 
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.update.append(optimizer.apply_gradients(zip(clipped_gradients, tf.trainable_variables())))

    elif self.mode == 'TEST':
      self.buckets = [(10, 50), (15, 50), (25, 50), (50, 50)] 

      self.outputs, self.losses = seq2seq.model_with_buckets(
           self.encoder_inputs, self.decoder_inputs, targets,
           self.target_weights, self.buckets, lambda x, y: seq2seq_multi(x, y, self.mode),
           softmax_loss_function = softmax_loss_function)
    
      for b in range(len(self.buckets)):
        #print('self.outputs[b]: ',self.outputs[b])
        self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1]
                           for output in self.outputs[b]]
        #print('self.outputs[b]: ',self.outputs[b])

    elif self.mode == 'RL':

      self.outputs, self.losses = seq2seq.model_with_buckets(
           self.encoder_inputs, self.decoder_inputs, targets,
           self.target_weights, self.buckets, lambda x, y: seq2seq_multi(x, y, self.mode),
           softmax_loss_function = softmax_loss_function, per_example_loss = True)
    
      #print('self.buckets: ',len(self.buckets))
      for b in range(len(self.buckets)):
        self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1]
                           for output in self.outputs[b]]

      #print('self.RL_index: ',self.RL_index)
      #print('self.outputs: ',len(self.outputs[0]),len(self.outputs[1]),len(self.outputs[2]),len(self.outputs[3]))
      #print('self.RL_index: ',len(self.RL_index))
      #print('self.outputs: ',len(self.outputs))
      for i, b in enumerate(self.outputs):
        prev_index = tf.multinomial(tf.log(tf.nn.softmax(b[self.buckets[i][1] - 1])), 1)
        #下面一行目的為補足最後一個decoder output，因為在decoder當中呼叫一次loop_function，RL_index才會append一次，但最後一個input得到的output不會再當prev丟入下一個loop_function，因此要從self.outputs的最後一個物件來補齊。
        self.RL_index[i] = tf.concat([self.RL_index[i], prev_index], axis = 1)
        #print(i,len(b))
        #print('self.buckets: ',self.buckets)
        #print('self.buckets[i][1]: ',self.buckets[i][1])
        #print('self.buckets[i][1] - 1: ',self.buckets[i][1] - 1)
        #print('b[self.buckets[i][1] - 1]: ', b[self.buckets[i][1] - 1])
        #print('prev_index: ',prev_index)
        #print('self.RL_index[i]: ',self.RL_index[i])
        #print('----------------')
      #self.outputs: list of 4 buckets, each (?,6258)
      #print('self.RL_index: ',self.RL_index)

      self.update = []
      optimizer = tf.train.GradientDescentOptimizer(0.01)
      #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in range(len(self.buckets)):
        scaled_loss = tf.multiply(self.losses[b], batch_reward)
        self.losses[b] = tf.reduce_mean(scaled_loss)
        gradients = tf.gradients(self.losses[b], tf.trainable_variables())
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.update.append(optimizer.apply_gradients(zip(clipped_gradients, tf.trainable_variables())))

    # specify saver
    self.saver = tf.train.Saver(max_to_keep = 2)

  # token_vector: list [batch_size, vocab_size] of length max_length
  # return: list of length batch_size, each contain the list of the decoded sentence
  def token2word(self, token_vector):
    sentence_list = [[] for _ in range(self.batch_size)]
  
    for logit in token_vector:
      outputs = np.argmax(logit, axis = 1)
      for i in range(self.batch_size):
        sentence_list[i].append(outputs[i])

    for i in range(self.batch_size):
      if data_utils.EOS_ID in sentence_list[i]:
        sentence_list[i] = sentence_list[i][:sentence_list[i].index(data_utils.EOS_ID)]
      if data_utils.PAD_ID in sentence_list[i]:
        sentence_list[i] = sentence_list[i][:sentence_list[i].index(data_utils.PAD_ID)]
      sentence_temp = [tf.compat.as_str(self.vocab_list[output]) for output in sentence_list[i]]  
      sentence_list[i] = " ".join(word for word in sentence_temp)
    return sentence_list

  # decoding function for reinforcement learning sampling output
  def token2word_RL(self, token_vector):
    sentence_list = [[] for _ in range(self.batch_size)]

    for i in range(self.batch_size):
      token_list = list(token_vector[i])
      if data_utils.EOS_ID in token_list:
        sentence_list[i] = token_list[:token_list.index(data_utils.EOS_ID)]

      sentence_temp = [tf.compat.as_str(self.vocab_list[output]) for output in sentence_list[i]]
      sentence_list[i] = " ".join(word for word in sentence_temp)
    return sentence_list

  # calculate logP(b|a)
  # a and b are both list of token ids. ex:[1,2,3,4,5...]
  # a--> encoder_input, b--> decoder_input in get_batch
  def prob(self, a, b, X, bucket_id):
    # define softmax
    def softmax(x):
      e_x = np.exp(x)
      return e_x / e_x.sum()

    # function X, not trainable, batch = 1
    temp = self.batch_size
    self.batch_size = 1
    encoder_input, decoder_input, weight = self.get_batch({bucket_id: [(a, b)]}, bucket_id)
    self.batch_size = temp
    outputs = X(encoder_input, decoder_input, weight, bucket_id)
    #print('b: ',b)
    #print('outputs: ',outputs,outputs[0].shape)
    r = 0.0
    # outputs已經project過(6258維)，看decoder_input的tokan_id(b)在output的softmax之機率高不高，越高reward越好。
    for logit, i in zip(outputs, b):
      #print('logit: ',logit,len(logit),logit[0].shape)
      #print('i: ',i)
      #print('r: ',np.log10(softmax(logit[0])[i]))
      r += np.log10(softmax(logit[0])[i])
    return r

  # this function is specify for training of Reinforcement Learning case
  def RL_readmap(self, map_path):
    self.vocab_dict, self.vocab_list = data_utils.read_map(map_path)

  def run(self, sess, encoder_inputs, decoder_inputs, target_weights,
          bucket_id, forward_only = False, X = None, Y = None):
    
    if self.mode == 'TEST':
        encoder_size = self.buckets[bucket_id][0]
        decoder_size = self.buckets[-1][-1] 
        decoder_inputs = np.reshape(np.repeat(decoder_inputs[0],decoder_size),(-1,1))
        target_weights = np.reshape(np.repeat(target_weights[0],decoder_size),(-1,1))
        print('decoder_inputs: ',len(decoder_inputs))
    else:
        encoder_size, decoder_size = self.buckets[bucket_id]
    #print('bucket_id: ',bucket_id)
    #print('encoder_size: ',encoder_size)
    #print('decoder_size: ',decoder_size)
    
    input_feed = {}
    for l in range(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in range(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype = np.int32)

    if self.mode == 'MLE':
      if forward_only:
        output_feed = [self.losses[bucket_id], self.outputs[bucket_id]]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]
      else:
        output_feed = [self.losses[bucket_id], self.update[bucket_id]]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    elif self.mode == 'TEST':
      output_feed = [self.outputs[bucket_id]]
      outputs = sess.run(output_feed, input_feed)
      return outputs[0]
    elif self.mode == 'RL':
      # check mode: sample or from decoder input
      # True for sample and False for from decoder input
      input_feed[self.loop_or_not] = True
      # input_feed == {..., <tf.Tensor 'Placeholder:0' shape=<unknown> dtype=bool>: True}

      # step 1: get seq2seq sampled output
      output_feed = [self.RL_index[bucket_id]]
      # output_feed == [<tf.Tensor 'concat:0' shape=(?, 10) dtype=int64>]
      outputs = sess.run(output_feed, input_feed)
      # outputs == batch_size list of token
      # sentence_rl is a batch sized list of sampled decoded natural sentence
      #sentence_rl = self.token2word_RL(outputs[0])
      #for a in sentence_rl:
      #  print(a)
      # step 2: get rewards according to some rules
      reward = np.ones((self.batch_size), dtype = np.float32)
      new_data = []
      for i in range(self.batch_size):
        token_ids = list(outputs[0][i])
        # token_ids是tf.multinomial取樣出來的東西
        if data_utils.EOS_ID in token_ids:
          token_ids = token_ids[:token_ids.index(data_utils.EOS_ID)]
        new_data.append(([], token_ids + [data_utils.EOS_ID]))
        '''
        # in this case, X is language model score
        # reward 1: ease of answering
        temp_reward = [self.prob(token_ids, data_utils.convert_to_token(tf.compat.as_bytes(sen), self.vocab_dict,
                       False) + [data_utils.EOS_ID], X, bucket_id)/float(len(sen)) for sen in self.dummy_reply]

        r1 = -np.mean(temp_reward)
        '''
        # reward 2: semantic coherence
        r_input = list(reversed([o[i] for o in encoder_inputs]))
        if data_utils.PAD_ID in r_input:
          r_input = r_input[:r_input.index(data_utils.PAD_ID)]

        r2 = self.prob(r_input, token_ids, X, bucket_id) / float(len(token_ids)) if len(token_ids) != 0 else 0

        # reward 3: sentiment analysis score
        #print('self.vocablist:' ,len(self.vocab_list))
        #print('token_ids: ',token_ids)
        word_token = []
        for token in token_ids:
            if token in self.vocab_list:
                word_token.append(self.vocab_list[token].decode('utf-8'))
        #word_token = [self.vocab_list[token].decode('utf-8') for token in token_ids]
        r3 = Y(word_token, np.array([len(token_ids)], dtype = np.int32))
        '''
        print('r1: %s' % r1)
        print('r2: %s' % r2)
        print('r3: %s' % r3)
        '''
        #reward[i] = 0.7 * r1 + 0.7 * r2 + r3
        reward[i] = 0 * r2 + r3
      #print(reward)
      # advantage
      reward = reward - np.mean(reward)
      _, decoder_inputs, target_weights = self.get_batch({bucket_id: new_data}, bucket_id, order = True)

      # step 3: update seq2seq model
      for l in range(decoder_size):
        input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        input_feed[self.target_weights[l].name] = target_weights[l]

      input_feed[self.reward] = reward
      input_feed[self.loop_or_not] = False
      output_feed = [self.losses[bucket_id], self.update[bucket_id]]
      #output_feed = [self.losses[bucket_id]]
      outputs = sess.run(output_feed, input_feed)

      return outputs[0]


  def get_batch(self, data, bucket_id, rand = True, order = False):
    # data should be [whole_data_length x (source, target)] 
    # decoder_input should contain "GO" symbol and target should contain "EOS" symbol
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # data[bucket_id] == [(incoder_inp_list,decoder_inp_list),...]
    encoder_input, decoder_input = random.choice(data[bucket_id])
    c = 0

    for i in range(self.batch_size):
      if rand:
        encoder_input, decoder_input = random.choice(data[bucket_id])
      if order:
        encoder_input, decoder_input = data[bucket_id][i]
        c += 1 

      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      decoder_pad = [data_utils.PAD_ID] * (decoder_size - len(decoder_input) - 1)
      decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    for length_idx in range(encoder_size):
      batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                                  for batch_idx in range(self.batch_size)], dtype = np.int32))

    for length_idx in range(decoder_size):
      batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                                  for batch_idx in range(self.batch_size)], dtype = np.int32))

      batch_weight = np.ones(self.batch_size, dtype = np.float32)
      for batch_idx in range(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

if __name__ == '__main__':

  test = Seq2seq(50, 100, 200, 300, 1, 128) 
