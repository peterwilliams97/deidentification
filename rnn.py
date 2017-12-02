import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from os.path import expanduser
from glob import glob
import re
import random
import time


# Parameters
MAX_VOCAB = 40000
SMALL_TEXT = True
learning_rate = 0.001
training_iters = 2000
display_step = 1000
n_input = 3
batch_size = 100
train_frac = .1

# number of units in RNN cell
n_hidden = 512

UNKNOWN = '<UNKNOWN>'

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


text = make_text()
print('%d bytes' % len(text))
all_words = RE_SPACE.split(text)
all_words = [w for w in all_words if len(w) > 1 and not RE_NUMBER.search(w)]
n = int(len(all_words) * (1.0 - train_frac))
# train = all_words[:n]
# test = all_words[n:]
# all_words = train
print('%d all_words n=%d' % (len(all_words), n))
print(' '.join(all_words[:100]))
vocabulary = sorted(set(all_words), key=lambda x: (len(x), x))
print('vocabulary=%d' % len(vocabulary))
print(vocabulary[:20])
word_counts = {w: 0 for w in vocabulary}
for w in all_words:
    word_counts[w] += 1
vocabulary_list = sorted(word_counts, key=lambda x: (-word_counts[x], x))[:MAX_VOCAB]
vocabulary_list[-1] = UNKNOWN
vocabulary = set(vocabulary_list)
for i, w in enumerate(all_words[:5]):
    marker = '***' if w in vocabulary else ''
    print('%3d: %-20s %s' % (i, w, marker))

vocab_size = len(vocabulary)
unk_index = vocab_size - 1
word_index = {w: i for i, w in enumerate(vocabulary_list)}
word_index[UNKNOWN] = unk_index
index_word = {i: w for w, i in word_index.items()}


for i in sorted(index_word)[:5]:
    print(i, index_word[i], type(i))

for i in range(vocab_size):
    assert i in index_word, i

embeddings = {w: np.zeros(vocab_size, dtype=np.float32) for w in vocabulary}
for w, i in word_index.items():
    embeddings[w][i] = 1.0
unk_embedding = embeddings[UNKNOWN]

print('vocabulary=%d' % len(vocabulary))
v_in, v_out = [], []
for w in all_words:
    if w in vocabulary:
        v_in.append(w)
    else:
        v_out.append(w)
print(' in vocabulary', len(v_in), len(set(v_in)))
print('out vocabulary', len(v_out), len(set(v_out)))


def data_getter(n_input):
    """Generator that returns  x, y
        Adds some randomness on selection process.
    """
    while True:
        i = random.randint(0, len(all_words) - n_input - 1)
        phrase = all_words[i:i + n_input + 1]
        wx = [word_index.get(phrase[j], unk_index) for j in range(n_input)]
        wy = word_index.get(phrase[n_input], unk_index)

        oneh_y = embeddings.get(phrase[n_input], unk_embedding)
        words_x = np.array(wx)
        words_x = np.reshape(words_x, [-1, 1])
        words_y = np.array(wy)
        yield oneh_y, words_x, words_y


def batch_getter(n_input, batch_size):
    """Generator that returns batches of x, y
        Adds some randomness on selection process.
    """
    source = data_getter(n_input)
    while True:
        oneh_y = np.empty((batch_size, vocab_size), dtype=np.float32)
        words_x = np.empty((batch_size, n_input, 1), dtype=int)
        words_y = np.empty((batch_size), dtype=int)
        for i in range(batch_size):
            oh_y, w_x, w_y = next(source)
            oneh_y[i] = oh_y
            words_x[i] = w_x
            words_y[i] = w_y
        words_x = np.reshape(words_x, [-1, n_input, 1])
        yield oneh_y, words_x, words_y


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
            indexes = [int(words_x[0, i, 0]) for i in range(words_x.shape[1])]
            symbols_in = [index_word.get(i, UNKNOWN) for i in indexes]
            symbols_out = index_word.get(words_y[0], UNKNOWN)
            v = tf.argmax(onehot_pred, 1).eval()
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
            symbols_in_keys = [word_index.get(w, unk_index) for w in words]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence, index_word[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except KeyError:
            print("Word not in dictionary")
