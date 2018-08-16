import tensorflow as tf
import data_utils
import os

hidden_size = 512
num_layers = 4 
batch_size = 32
model_dir = 'model/xhj_ptt_%s_%s_%s/'%(hidden_size,num_layers,batch_size) 
model_RL_dir = 'model_RL/xhj_ptt_%s_%s_%s/'%(hidden_size,num_layers,batch_size)
if not os.path.exists(model_dir):
    print('create model dir: ',model_dir)
    os.mkdir(model_dir)
if not os.path.exists(model_RL_dir):
    print('create model RL dir: ',model_RL_dir)
    os.mkdir(model_RL_dir)

tf.app.flags.DEFINE_integer('vocab_size', data_utils.WORD_DIM, 'vocabulary size of the input')
tf.app.flags.DEFINE_integer('hidden_size', hidden_size, 'number of units of hidden layer')
tf.app.flags.DEFINE_integer('num_layers', num_layers, 'number of layers')
tf.app.flags.DEFINE_integer('batch_size', batch_size, 'batch size')
tf.app.flags.DEFINE_string('mode', 'MLE', 'mode of the seq2seq model')
tf.app.flags.DEFINE_string('source_data_dir', 'corpus/source', 'directory of source')
tf.app.flags.DEFINE_string('target_data_dir', 'corpus/target', 'directory of target')
tf.app.flags.DEFINE_string('model_dir', model_dir, 'directory of model')
tf.app.flags.DEFINE_string('model_rl_dir',model_RL_dir, 'directory of RL model')
tf.app.flags.DEFINE_integer('check_step', '300', 'step interval of saving model')
# for rnn dropout
tf.app.flags.DEFINE_float('input_keep_prob', '1.0', 'step input dropout of saving model')
tf.app.flags.DEFINE_float('output_keep_prob', '1.0', 'step output dropout of saving model')
tf.app.flags.DEFINE_float('state_keep_prob', '1.0', 'step state dropout of saving model')
# output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.
# beam search
tf.app.flags.DEFINE_boolean('beam_search', False, 'beam search')
tf.app.flags.DEFINE_integer('beam_size', 10 , 'beam size')
tf.app.flags.DEFINE_boolean('debug', True, 'debug')
# schedule sampling
tf.app.flags.DEFINE_string('schedule_sampling', 'inverse_sigmoid', 'schedule sampling type[linear|exp|inverse_sigmoid|False]')
tf.app.flags.DEFINE_float('sampling_decay_rate', 0.99 , 'schedule sampling decay rate')
tf.app.flags.DEFINE_integer('sampling_global_step', 10000, 'sampling_global_step')
tf.app.flags.DEFINE_integer('sampling_decay_steps', 5000, 'sampling_decay_steps')


FLAGS = tf.app.flags.FLAGS
