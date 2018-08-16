import tensorflow as tf
import numpy as np
import json
import re
import os
import sys 
sys.path.append('sentiment_analysis/')
import math
from termcolor import colored

import data_utils
import seq2seq_model
from sentiment_analysis import run
from sentiment_analysis import dataset
from flags import FLAGS

with open('replace_words.json','r') as f:
   replace_words = json.load(f) 

SEED = 112

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
buckets = [(10, 10), (15, 15), (25, 25), (50, 50)]

def sub_words(word):
    for rep in replace_words.keys():
        if rep in word:
            word = re.sub(rep,replace_words[rep],word)
    return word

# mode variable has three different mode:
# 1. MLE
# 2. RL
# 3. TEST
def create_seq2seq(session, mode):

  if mode == 'TEST':
    FLAGS.schedule_sampling = False 
  else:
    FLAGS.beam_search = False
  print('FLAGS.beam_search: ',FLAGS.beam_search)
  if FLAGS.beam_search:
    print('FLAGS.beam_size: ',FLAGS.beam_size)
    print('FLAGS.debug: ',bool(FLAGS.debug))
      
  model = seq2seq_model.Seq2seq(vocab_size = FLAGS.vocab_size,
                                buckets = buckets,
                                size = FLAGS.hidden_size,
                                num_layers = FLAGS.num_layers,
                                batch_size = FLAGS.batch_size,
                                mode = mode,
                                input_keep_prob = FLAGS.input_keep_prob,
                                output_keep_prob = FLAGS.output_keep_prob,
                                state_keep_prob = FLAGS.state_keep_prob,
                                beam_search = FLAGS.beam_search,
                                beam_size = FLAGS.beam_size,
                                schedule_sampling = FLAGS.schedule_sampling,
                                sampling_decay_rate = FLAGS.sampling_decay_rate,
                                sampling_global_step = FLAGS.sampling_global_step,
                                sampling_decay_steps = FLAGS.sampling_decay_steps
                                )
  
  #if mode != 'TEST':
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  #else:
  #  ckpt = tf.train.get_checkpoint_state(FLAGS.model_rl_dir)
  
  if ckpt:
    print("Reading model from %s, mode: %s" % (ckpt.model_checkpoint_path, mode))
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters, mode: %s" % mode)
    session.run(tf.global_variables_initializer())
  
  return model

def train_MLE(): 
  data_utils.prepare_whole_data(FLAGS.source_data_dir, FLAGS.target_data_dir, FLAGS.vocab_size)

  # read dataset and split to training set and validation set
  d = data_utils.read_data(FLAGS.source_data_dir + '.token', FLAGS.target_data_dir + '.token', buckets)
  np.random.seed(SEED)
  np.random.shuffle(d)
  print('Total document size: %s' % sum(len(l) for l in d))
  print('len(d): ', len(d))
  d_train = [[] for _ in range(len(d))]
  d_valid = [[] for _ in range(len(d))]
  for i in range(len(d)):
    d_train[i] = d[i][:int(0.9 * len(d[i]))]
    d_valid[i] = d[i][int(-0.1 * len(d[i])):]

  train_bucket_sizes = [len(d_train[b]) for b in range(len(d_train))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]
  print('train_bucket_sizes: ',train_bucket_sizes)
  print('train_total_size: ',train_total_size)
  print('train_buckets_scale: ',train_buckets_scale)
  valid_bucket_sizes = [len(d_valid[b]) for b in range(len(d_valid))]
  valid_total_size = float(sum(valid_bucket_sizes))
  valid_buckets_scale = [sum(valid_bucket_sizes[:i + 1]) / valid_total_size
                         for i in range(len(valid_bucket_sizes))]
  print('valid_bucket_sizes: ',valid_bucket_sizes)
  print('valid_total_size: ',valid_total_size)
  print('valid_buckets_scale: ',valid_buckets_scale)

  with tf.Session() as sess:

    model = create_seq2seq(sess, 'MLE')
    step = 0
    loss = 0
    loss_list = []
 
    print('sampling_decay_steps: ',FLAGS.sampling_decay_steps)
    print('sampling_probability: ',sess.run(model.sampling_probability))
    print('-----')
    while(True):
      step += 1

      random_number = np.random.random_sample()
      # buckets_scale 是累加百分比
      bucket_id = min([i for i in range(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number])
      encoder_input, decoder_input, weight = model.get_batch(d_train, bucket_id)
      #print('batch_size: ',model.batch_size)      ==> 64
      #print('batch_size: ',len(encoder_input[0])) ==> 64
      #print('batch_size: ',len(encoder_input))    ==> 15,50,...
      #print('batch_size: ',len(decoder_input))    ==> 15,50,... 
      #print('batch_size: ',len(weight))           ==> 15,50,...
      loss_train, _ = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
      loss += loss_train / FLAGS.check_step
      #print(model.token2word(sen)[0])
      if step!=0 and step % FLAGS.sampling_decay_steps == 0:
        sess.run(model.sampling_probability_decay)
        print('sampling_probability: ',sess.run(model.sampling_probability))
        if_feed_prev = bernoulli_sampling(model.sampling_probability)
        if_feed_prev = sess.run(if_feed_prev)
        print('if_feed_prev: ',not if_feed_prev)
        
      if step % FLAGS.check_step == 0:
        print('Step %s, Training perplexity: %s, Learning rate: %s' % (step, math.exp(loss),
                                  sess.run(model.learning_rate))) 
        for i in range(len(d)):
          encoder_input, decoder_input, weight = model.get_batch(d_valid, i)
          loss_valid, _ = model.run(sess, encoder_input, decoder_input, weight, i, forward_only = True)
          print('  Validation perplexity in bucket %s: %s' % (i, math.exp(loss_valid)))
        if len(loss_list) > 2 and loss > max(loss_list[-3:]):
          sess.run(model.learning_rate_decay)
        loss_list.append(loss)  
        loss = 0

        checkpoint_path = os.path.join(FLAGS.model_dir, "MLE.ckpt")
        model.saver.save(sess, checkpoint_path, global_step = step)
        print('Saving model at step %s' % step)

def train_RL():
  g1 = tf.Graph()
  g2 = tf.Graph()
  g3 = tf.Graph()
  sess1 = tf.Session(graph = g1)
  sess2 = tf.Session(graph = g2)
  sess3 = tf.Session(graph = g3)
  # model is for training seq2seq with Reinforcement Learning
  with g1.as_default():
    model = create_seq2seq(sess1, 'RL')
    # we set sample size = ?
    model.batch_size = 5
  # model_LM is for a reward function (language model)
  with g2.as_default():
    model_LM = create_seq2seq(sess2, 'MLE')
    model_LM.beam_search = False
    # calculate probibility of only one sentence
    model_LM.batch_size = 1

  def LM(encoder_input, decoder_input, weight, bucket_id):
    return model_LM.run(sess2, encoder_input, decoder_input, weight, bucket_id, forward_only = True)[1]
  # new reward function: sentiment score
  with g3.as_default():
    model_SA = run.create_model(sess3, 'test') 
    model_SA.batch_size = 1
 
  def SA(sentence, encoder_length):
    sentence = ' '.join(sentence)
    token_ids = dataset.convert_to_token(sentence, model_SA.vocab_map)
    encoder_input, encoder_length, _ = model_SA.get_batch([(0, token_ids)])
    return model_SA.step(sess3, encoder_input, encoder_length)[0][0]

  data_utils.prepare_whole_data(FLAGS.source_data_dir, FLAGS.target_data_dir, FLAGS.vocab_size)
  d = data_utils.read_data(FLAGS.source_data_dir + '.token', FLAGS.target_data_dir + '.token', buckets)

  train_bucket_sizes = [len(d[b]) for b in range(len(d))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]

  # make RL object read vocab mapping dict, list  
  model.RL_readmap(FLAGS.source_data_dir + '.' + str(FLAGS.vocab_size) + '.mapping')
  step = 0
  while(True):
    step += 1

    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number])
    
    # the same encoder_input for sampling batch_size times
    #encoder_input, decoder_input, weight = model.get_batch(d, bucket_id, rand = False)    
    encoder_input, decoder_input, weight = model.get_batch(d, bucket_id, rand = False)    
    loss = model.run(sess1, encoder_input, decoder_input, weight, bucket_id, X = LM, Y = SA)
   
    # debug 
    #encoder_input = np.reshape(np.transpose(encoder_input, (1, 0, 2)), (-1, FLAGS.vocab_size))
    #encoder_input = np.split(encoder_input, FLAGS.max_length)

    #print(model.token2word(encoder_input)[0])
    #print(model.token2word(sen)[0])
    
    if step % FLAGS.check_step == 0:
      print('Loss at step %s: %s' % (step, loss))
      checkpoint_path = os.path.join('model_RL', "RL.ckpt")
      model.saver.save(sess1, checkpoint_path, global_step = step)
      print('Saving model at step %s' % step)


def test():
  sess = tf.Session()
  vocab_dict, vocab_list = data_utils.read_map(FLAGS.source_data_dir + '.' + str(FLAGS.vocab_size) + '.mapping')
  model = create_seq2seq(sess, 'TEST')
  model.batch_size = 1
  
  sys.stdout.write("Input sentence: ")
  sys.stdout.flush()
  sentence = sys.stdin.readline()
  sentence = (' ').join([s for s in sentence])

  while(sentence):
    token_ids = data_utils.convert_to_token(tf.compat.as_bytes(sentence), vocab_dict, False)
    bucket_id = len(buckets) - 1
    for i, bucket in enumerate(buckets):
      if bucket[0] >= len(token_ids):
        bucket_id = i
        break
    # Get a 1-element batch to feed the sentence to the model.
    encoder_input, decoder_input, weight = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    output = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    
    # beam search all
    if bool(model.beam_search) is True:
        if bool(FLAGS.debug):
            outs = []
            for _ in range(model.beam_size):
                outs.append([])
   
            for out in output:
                for i,o in enumerate(out):
                    outs[i].append(o)
            outs = np.array(outs)
            #print('outs: ',outs.shape)
            outputss = []
            for out in outs:
                #print('out: ',out.shape)
                outputs = [int(np.argmax(logit)) for logit in out]
                outputss.append(outputs)
    
            for i,outputs in enumerate(outputss):
                sys_reply = "".join([tf.compat.as_str(vocab_list[output]) for output in outputs])
                sys_reply = sub_words(sys_reply)
                if i == 0:
                    print(colored("Syetem reply(bs best): " + sys_reply,"red"))
                else:
                    print("Syetem reply(bs all): " + sys_reply)
        else:
            output = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output]
            if data_utils.EOS_ID in outputs:
              outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            sys_reply = "".join([tf.compat.as_str(vocab_list[output]) for output in outputs])
            sys_reply = sub_words(sys_reply)
            print("Syetem reply(bs best): " + sys_reply)
            

    # MLE
    else:
        output = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
        print('output: ', len(output), output[0].shape)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output]
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
          outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        sys_reply = "".join([tf.compat.as_str(vocab_list[output]) for output in outputs])
        sys_reply = sub_words(sys_reply)
        print("Syetem reply(MLE): " + sys_reply)


    # Print out French sentence corresponding to outputs.
    #print("Syetem reply: " + "".join([tf.compat.as_str(vocab_list[output]) for output in outputs]))
    print("User input  : ", end="")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    sentence = (' ').join([s for s in sentence])

if __name__ == '__main__':
  if FLAGS.mode == 'MLE':
    train_MLE()
  elif FLAGS.mode == 'RL':
    train_RL()
  else:
    test()


