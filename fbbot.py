from flask import Flask, request
from fb_setting import *
import requests

# ai
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('sentiment_analysis/')
import math
import data_utils
import seq2seq_model
from sentiment_analysis import run
from sentiment_analysis import dataset
from flags import FLAGS 
from run import replace_words,SEED,buckets,sub_words,create_seq2seq
from util import *


sess = tf.Session()
vocab_dict, vocab_list = data_utils.read_map(FLAGS.source_data_dir + '.' + str(FLAGS.vocab_size) + '.mapping')
model = create_seq2seq(sess, 'TEST')
model.batch_size = 1

# create a Flask app instance
app = Flask(__name__)

# method to reply to a message from the sender
def reply(user_id, msg):
    data = {
        "recipient": {"id": user_id},
        "message": {"text": msg}
    }
    # Post request using the Facebook Graph API v2.6
    resp = requests.post("https://graph.facebook.com/v3.1/me/messages?access_token=" + ACCESS_TOKEN, json=data)
    print(resp.content)

# GET request to handle the verification of tokens
@app.route('/', methods=['GET'])
def handle_verification():
    if request.args['hub.verify_token'] == VERIFY_TOKEN:
        return request.args['hub.challenge']
    else:
        return "Invalid verification token"

# POST request to handle in coming messages then call reply()
@app.route('/', methods=['POST'])
def handle_incoming_messages():
    data = request.json
    sender = data['entry'][0]['messaging'][0]['sender']['id']
    message = data['entry'][0]['messaging'][0]['message']['text']
    sentence = (' ').join([s for s in message])
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
    outputs = [int(np.argmax(logit, axis=1)) for logit in output]
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
      outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    message = "".join([tf.compat.as_str(vocab_list[output]) for output in outputs])
    message = sub_words(message)
    message = qulify_sentence(message)
    reply(sender, message)
    return "ok"

# Run the application.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
