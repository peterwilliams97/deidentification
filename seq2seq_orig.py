# coding: utf-8
"""
# # Date Converter
#
# We will be translating from one date format to another. In order to do this we need to connect two
# set of LSTMs (RNNs). The diagram looks as follows: Each set respectively sharing weights (i.e.
# each of the 4 green cells have the same weights and similarly with the blue cells). The first is a
# many-to-one LSTM, which summarises the question at the last hidden layer (and cell memory).
#
# The second set (blue) is a many-to-many LSTM which has different weights to the first set of LSTMs.
# The input is simply the answer sentence while the output is the same sentence shifted by one. Of
# course during testing time there are no inputs for the `answer` and it is only used during training.
# ![seq2seq_diagram](https://i.stack.imgur.com/YjlBt.png)
#
# **20th January 2017 => 20th January 2009**
# ![troll](./images/troll_face.png)
#
# ## References:
# 1. Plotting Tensorflow graph:
    https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter/38192374#38192374
# 2. The generation process was taken from: https://github.com/datalogue/keras-attention/blob/master/data/generate.py
# 3. 2014 paper with 2000+ citations: https://arxiv.org/pdf/1409.3215.pdf
"""
import numpy as np
import random
import time
from faker import Faker
import babel
from babel.dates import format_date
import tensorflow as tf
from utilities import show_graph
from sklearn.model_selection import train_test_split


def describe(x):
    try:
        v = '%s:%s' % (list(x.shape), x.dtype)
    except AttributeError:
        if isinstance(x, (list, tuple)):
            v = '%d:%s' % (len(x), type(x[0]))
        else:
            v = type(x)
    return v


def show(name, x):
    if isinstance(x, (list, tuple)):
        v = '%d:%s' % (len(x), describe(x))
    else:
        v = describe(x)
    print('>>>"%s"=%s' % (name, v))


fake = Faker()
fake.seed(42)
random.seed(42)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY',
           ]

# change this if you want it to work with only a single language
LOCALES = babel.localedata.locale_identifiers()
LOCALES = [lang for lang in LOCALES if 'en' in str(lang)]


def create_date():
    """
        Creates some fake dates
        :returns: tuple containing
                  1. human formatted string
                  2. machine formatted string
                  3. date object.
    """
    dt = fake.date_object()

    # wrapping this in a try catch because
    # the locale 'vo' and format 'full' will fail
    try:
        human = format_date(dt,
                            format=random.choice(FORMATS),
                            locale=random.choice(LOCALES))

        case_change = random.randint(0, 3)  # 1/2 chance of case change
        if case_change == 1:
            human = human.upper()
        elif case_change == 2:
            human = human.lower()

        machine = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human, machine  # , dt


data = [create_date() for _ in range(50000)]


# See below what we are trying to do in this lesson. We are taking dates of various formats and
# converting them into a standard date format:

print('data=%s' % data[:5])

x = [x for x, y in data]
y = [y for x, y in data]

u_characters = set(' '.join(x))
char2numX = dict(zip(u_characters, range(len(u_characters))))

u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))

print(x[:3])
print(y[:3])
print(sorted(char2numX.items())[:3])
print(sorted(char2numY.items())[:3])

# Pad all sequences that are shorter than the max length of the sequence
char2numX['<PAD>'] = len(char2numX)
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
max_len = max([len(date) for date in x])

x = [[char2numX['<PAD>']] * (max_len - len(date)) + [char2numX[x_] for x_ in date] for date in x]
print(''.join([num2charX[x_] for x_ in x[4]]))
x = np.array(x)

char2numY['<GO>'] = len(char2numY)
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
print(''.join([num2charY[y_] for y_ in y[4]]))
y = np.array(y)

x_seq_length = len(x[0])
y_seq_length = len(y[0]) - 1
print(x_seq_length, y_seq_length)
print(x[0])
print(y[0])
print([num2charX[i] for i in x[0]])
print([num2charY[i] for i in y[0]])


def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size


epochs = 10
batch_size = 128
nodes = 32
embed_size = 10

print('=' * 80)
print('epochs=%d' % epochs)
print('embed_size=%d' % embed_size)
print('batch_size=%d' % batch_size)
print('nodes=%d' % nodes)
print('x_seq_length=%d' % x_seq_length)
print('y_seq_length=%d' % y_seq_length)
print('len(char2numX=%d' % len(char2numX))
print('len(char2numY=%d' % len(char2numY))

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0),
                              name='enc_embedding')
show('input_embedding', input_embedding)
# TODO: create the variable output embedding
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0),
                               name='dec_embedding')
show('output_embedding', output_embedding)

# TODO: Use tf.nn.embedding_lookup to complete the next two lines
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)
show('date_input_embed', date_input_embed)
show('date_output_embed', date_output_embed)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

with tf.variable_scope("decoding") as decoding_scope:
    # TODO: create the decoder LSTMs, this is very similar to the above
    # you will need to set initial_state=last_state from the encoder
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed,
                                       initial_state=last_state)
# connect outputs to
# len(char2numY)=13
logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(char2numY),
                                           activation_fn=None)
with tf.name_scope("optimization"):
    # Loss function logits and labels must have the same first dimension,
    # got logits shape [3712,13] and labels shape [1280]
    show('logits', logits)
    show('targets', targets)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

print('dec_outputs=%s' % dec_outputs.get_shape().as_list())
print('last_state[0]=%s' % last_state[0].get_shape().as_list())
print('inputs=%s' % inputs.get_shape().as_list())
print('date_input_embed=%s' % date_input_embed.get_shape().as_list())


# Train the graph
show_graph(tf.get_default_graph().as_graph_def())

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

sess.run(tf.global_variables_initializer())
print('@' * 80)
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        print('batch_i=%d' % (batch_i))
        show('inputs', source_batch)
        show('outputs', target_batch[:, :-1])
        show('targets', target_batch[:, 1:])
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
            feed_dict={inputs: source_batch,
                       outputs: target_batch[:, :-1],
                       targets: target_batch[:, 1:]})
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(
          epoch_i, batch_loss,
          accuracy, time.time() - start_time))


# Translate on test set

# In[20]:

source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))

dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                feed_dict = {inputs: source_batch,
                 outputs: dec_input})
    prediction = batch_logits[:,-1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:,None]])

print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))


# Let's randomly take two from this test set and see what it spits out:

# In[21]:

num_preds = 2
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in)+' => '+''.join(date_out))


# In[ ]:



