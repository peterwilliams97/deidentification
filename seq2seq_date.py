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

    last_state is the last state of the encoder LSTM  lstm_enc
                      initial state of the decoder LSTM lstm_dec
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


test_size = 0.3
epochs = 200
batch_size = 128
hidden_size = 4  # Number of dimensions of cell 32 works well
embed_size = 20   # Character embedding size     3-10 works well
patience = 5      # Number of epochs in which test accuracy doesn't increase after which we give up


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


def create_dates():
    """
        Creates some fake dates
        returns: tuple containing
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
        return None, None

    return human, machine


data = [create_dates() for _ in range(50000)]


# See below what we are trying to do in this lesson. We are taking dates of various formats and
# converting them into a standard date format:

print('data=%s' % data[:5])

x, y = zip(*data)

char2numX = {c: i for i, c in enumerate(sorted(set().union(*x)))}
char2numY = {c: i for i, c in enumerate(sorted(set().union(*y)))}

print('x:', x[:3])
for i, (c, n) in enumerate(sorted(char2numX.items())[:5]):
    print('%5d: %3d %s' % (i, n, c))
print('y:', y[:3])
for i, (c, n) in enumerate(sorted(char2numY.items())[:5]):
    print('%5d: %3d %s' % (i, n, c))

# Pad all sequences that are shorter than the max length of the sequence
char2numX['<PAD>'] = len(char2numX)
num2charX = {i: c for c, i in char2numX.items()}
max_len = max([len(date) for date in x])

x = [[char2numX['<PAD>']] * (max_len - len(date)) + [char2numX[c] for c in date] for date in x]
print(''.join([num2charX[x_] for x_ in x[4]]))
x = np.array(x)

char2numY['<GO>'] = len(char2numY)
num2charY = {i: c for c, i in char2numY.items()}

y = [[char2numY['<GO>']] + [char2numY[c] for c in date] for date in y]
print(''.join([num2charY[c] for c in y[4]]))
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
    for start in range(0, len(x) - batch_size + 1, batch_size):
        yield x[start:start + batch_size], y[start:start + batch_size]


print('=' * 80)
print('epochs=%d' % epochs)
print('embed_size=%d' % embed_size)
print('batch_size=%d' % batch_size)
print('hidden_size=%d' % hidden_size)
print('x_seq_length=%d' % x_seq_length)
print('y_seq_length=%d' % y_seq_length)
print('len(char2numX=%d' % len(char2numX))
print('len(char2numY=%d' % len(char2numY))

#
# Build the computation graph
#
tf.reset_default_graph()
sess = tf.Session()

# Tensor where we will feed the data into graph
# Can't specify outputs size (y_seq_length) because we build it from partial strings in predict()
inputs = tf.placeholder(tf.int32, (batch_size, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (batch_size, None), 'output')
targets = tf.placeholder(tf.int32, (batch_size, y_seq_length), 'targets')

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0),
                              name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0),
                               name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

show('input_embedding', input_embedding)
show('output_embedding', output_embedding)
show('date_input_embed', date_input_embed)
show('date_output_embed', date_output_embed)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

with tf.variable_scope("decoding") as decoding_scope:
    # decoder initial_state=last_state from the encoder. !@#$ Is this "weight-sharing"?
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed,
                                       initial_state=last_state)

# logits of decoder output
logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(char2numY),
                                           activation_fn=None)
with tf.name_scope("optimization"):
    show('logits', logits)
    show('targets', targets)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

print('dec_outputs=%s' % dec_outputs.get_shape().as_list())
print('last_state[0]=%s' % last_state[0].get_shape().as_list())
print('inputs=%s' % inputs.get_shape().as_list())
print('date_input_embed=%s' % date_input_embed.get_shape().as_list())


def predict(source_batch):
    dec_input = np.ones((len(source_batch), 1)) * char2numY['<GO>']

    # Build y sequence left to right
    for _ in range(y_seq_length):
        batch_logits = sess.run(logits,
                                feed_dict={inputs: source_batch,
                                           outputs: dec_input})
        prediction = batch_logits[:, -1].argmax(axis=-1)
        dec_input = np.hstack([dec_input, prediction[:, None]])

    return dec_input


def evaluate(source_batch, target_batch, dec_input):
    dec_input = predict(source_batch)
    correct = np.sum(dec_input == target_batch)
    total = target_batch.size
    return total, correct


def demonstrate(source_batch, dec_input, num_preds):
    # Randomly take `num_preds` from the test set and see what it spits out:
    # !@#$ shuffe
    source_sents = source_batch[:num_preds]
    dest_sents = dec_input[:num_preds, 1:]

    results = []
    for src, dst in zip(source_sents, dest_sents):
        src_chars = [num2charX[l] for l in src if num2charX[l] != "<PAD>"]
        dst_chars = [num2charY[l] for l in dst]
        src_text = ''.join(src_chars)
        dst_text = ''.join(dst_chars)
        results.append((src_text, dst_text))

    return results


def predict_eval(num_preds=0):
    start_time = time.time()
    total, correct = 0, 0
    results = []
    for source_batch, target_batch in batch_data(X_test, y_test, batch_size):
        dec_input = predict(source_batch)
        tot, corr = evaluate(source_batch, target_batch, dec_input)
        total += tot
        correct += corr
        r = demonstrate(source_batch, dec_input, max(0, num_preds - len(results)))
        if r:
            results.extend(r)
            results = results[:num_preds]
    acc = correct / total
    return acc, results, time.time() - start_time


# Train the graph
show_graph(tf.get_default_graph().as_graph_def())


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

sess.run(tf.global_variables_initializer())

print('@' * 80)
best_epoch = -1
best_accuracy = 0.0

for epoch_i in range(epochs):
    total_loss = 0.0
    total_accuracy = 0.0
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
            feed_dict={inputs: source_batch,
                       outputs: target_batch[:, :-1],
                       targets: target_batch[:, 1:]})
        accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
        total_loss += batch_loss
        total_accuracy += accuracy

    train_loss = total_loss / (batch_i + 1)
    train_accuracy = total_accuracy / (batch_i + 1)
    train_duration = time.time() - start_time
    test_accuracy, results, test_duration = predict_eval(5)

    improved = test_accuracy > best_accuracy
    if improved:
        best_epoch = epoch_i
        best_accuracy = test_accuracy
    print('Epoch %3d (best %3d) Loss: %6.3f Accuracy: %6.4f Test Accuracy %6.4f '
          'Epoch duration: %6.3f sec %6.3f sec (%d batches)' %
          (epoch_i, best_epoch, train_loss, train_accuracy, test_accuracy,
           train_duration, test_duration, batch_i + 1))
    if improved:
        for i, (src, dst) in enumerate(results):
            print('%6d: %20s => %s' % (i, src, dst))
    if epoch_i > best_epoch + patience:
        break
