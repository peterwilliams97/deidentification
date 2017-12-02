import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from os.path import expanduser
from glob import glob
import re
from itertools import cycle
import random
import time


# Parameters
MAX_VOCAB = 40000
SMALL_TEXT = False
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3
batch_size = 100

# number of units in RNN cell
n_hidden = 512


# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

start_time = time.time()


def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < 60 * 60:
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


def show(name, x):
    try:
        v = '%s:%s' % (list(x.shape), x.dtype)
    except AttributeError:
        if isinstance(x, (list, tuple)):
            v = '%d:%s' % (len(x), type(x[0]))
        else:
            v = type(x)
    print('"%s"=%s' % (name, v))


def read_text(path):
    with open(path, 'rt') as f:
        return f.read()


def make_text():
    if SMALL_TEXT:
        return read_text('belling_the_cat.txt')
    mask = expanduser('~/testdata.clean/deploy/*.txt')
    file_list = sorted(glob(mask))
    print('%d files' % len(file_list))
    return '\n'.join(read_text(path) for path in file_list)


RE_SPACE = re.compile(r'[\s,\.:;!\?\-\(\)]+', re.DOTALL | re.MULTILINE)
RE_NUMBER = re.compile(r'[^A-Z^a-z]')

train_frac = .1
text = make_text()
print('%d bytes' % len(text))
words = RE_SPACE.split(text)
words = [w for w in words if len(w) > 1 and not RE_NUMBER.search(w)]
n = int(len(words) * (1.0 - train_frac))
# train = words[:n]
# test = words[n:]
# words = train
print('%d words n=%d' % (len(words), n))
print(' '.join(words[:100]))
vocabulary = sorted(set(words), key=lambda x: (len(x), x))
print('vocabulary=%d' % len(vocabulary))
print(vocabulary[:20])
word_counts = {w: 0 for w in vocabulary}
for w in words:
    word_counts[w] += 1
vocabulary_list = sorted(word_counts, key=lambda x: (-word_counts[x], x))[:MAX_VOCAB]
vocabulary_list[-1] = '<UNKNOWN>'
vocabulary = set(vocabulary_list)
for i, w in enumerate(words[:5]):
    marker = '***' if w in vocabulary else ''
    print('%3d: %-20s %s' % (i, w, marker))

vocab_size = len(vocabulary)
word_index = {w: i for i, w in enumerate(vocabulary_list)}
index_word = {i: w for w, i in word_index.items()}

for i in sorted(index_word)[:5]:
    print(i, index_word[i], type(i))

for i in range(vocab_size):
    assert i in index_word, i

embeddings = {w: np.zeros(vocab_size, dtype=np.float32) for w in vocabulary}
for w, i in word_index.items():
    embeddings[w][i] = 1.0
zero = embeddings['<UNKNOWN>']
# np.zeros(vocab_size, dtype=np.float32)

print('vocabulary=%d' % len(vocabulary))
v_in, v_out = [], []
for w in words:
    if w in vocabulary:
        v_in.append(w)
    else:
        v_out.append(w)
print(' in vocabulary', len(v_in), len(set(v_in)))
print('out vocabulary', len(v_out), len(set(v_out)))

def _data_getter(n_input):
    source = cycle(words)
    batch = []
    for _ in range(n_input):
        w = next(source)
        i = word_index.get(w, -1)
        batch.append(i)
    while True:
        x = np.empty((n_input, vocab_size), dtype=np.float32)
        for i in range(n_input):
            x[i] = embeddings.get(batch[i], zero)
        w = next(source)
        i = word_index.get(w, -1)
        y = embeddings.get(w, zero)
        w_x = np.array(batch)
        w_x = np.reshape(w_x, [-1, 1])
        w_y = np.array(i)

        yield x, y, w_x, w_y
        batch = batch[1:] + [i]


def data_getter(n_input):
    while True:
        i = random.randint(0, len(words) - n_input - 1)
        phrase = words[i:i + n_input + 1]
        words_x = [word_index.get(phrase[j], -1) for j in range(n_input)]
        words_y = word_index.get(phrase[n_input], -1)

        # oneh_x = np.empty((n_input, vocab_size), dtype=np.float32)
        # for j in range(n_input):
        #     oneh_x[j] = embeddings.get(phrase[j], zero)
        oneh_y = embeddings.get(phrase[n_input], zero)
        w_x = np.array(words_x)
        w_x = np.reshape(w_x, [-1, 1])
        w_y = np.array(words_y)

        # is_zero = word_y not in embeddings
        # print('**', phrase, words_x, word_y, is_zero)

        yield oneh_y, w_x, w_y


def batch_getter(n_input, batch_size):
    source = data_getter(n_input)
    while True:
        # xx = np.empty((batch_size, n_input, vocab_size), dtype=np.float32)
        yy = np.empty((batch_size, vocab_size), dtype=np.float32)
        wxx = np.empty((batch_size, n_input, 1), dtype=int)
        wyy = np.empty((batch_size), dtype=int)
        for i in range(batch_size):
            y, wx, wy = next(source)
            yy[i] = y
            wxx[i] = wx
            wyy[i] = wy
            # wxx.append(wx)
            # wyy.append(wy)
        # xx = np.reshape(xx, [-1, n_input, vocab_size, 1])
        wxx = np.reshape(wxx, [-1, n_input, 1])
        yield yy, wxx, wyy


# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


def RNN(x, weights, biases):
    show('x in', x)

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])
    show('x reshaped', x)

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)
    show('x split', x)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

source = batch_getter(n_input, batch_size)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0, n_input + 1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    for step in range(training_iters):
        # Generate a minibatch. Add some randomness on selection process.
        batch_y, words_x, words_y = next(source)
        # xx, yy = words_x, batch_y
        # print(step)
        # show('batch_x', batch_x)
        # show('batch_y', batch_y)
        # show('words_x', words_x)
        # show('words_y', words_y)
        # assert step < 5

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred],
                                                feed_dict={x: words_x,
                                                           y: batch_y})
        loss_total += loss
        acc_total += acc
        if (step + 1) % display_step == 0:
            print("Iter=%6d: Average Loss=%9.6f, Average Accuracy=%5.2f%%" %
                  (step + 1, loss_total / display_step, 100.0 * acc_total / display_step))
            acc_total = 0
            loss_total = 0
            # show('words_x', words_x)
            # print(words_x)
            indexes = [int(words_x[0, i, 0]) for i in range(words_x.shape[1])]
            # print(indexes, type(indexes[0]))
            symbols_in = [index_word.get(i, '<UNKNOWN>') for i in indexes]
            # show('words_y', words_y)
            # print(words_y)
            symbols_out = index_word.get(words_y[0], '<UNKNOWN>')
            v = tf.argmax(onehot_pred, 1).eval()
            # show('v', v)
            # print(v)
            symbols_out_pred = index_word[int(v[0])]
            print("%s -> [%s] predicted [%s]" % (symbols_in,symbols_out, symbols_out_pred))

    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")
    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")
