import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from os.path import expanduser
from glob import glob
import re
import random
import time
import os
import spacy
from collections import defaultdict


# Parameters
seed = 112
use_spacy = True
MAX_VOCAB = 10000
MAX_WORDS = 10000
SMALL_TEXT = False
learning_rate = 0.001
n_input = 3
batch_size = 100
train_frac = .1
n_epochs = 1000
n_epochs_report = 10
patience = 20
model_folder = 'models'

# number of units in RNN cell
n_hidden = 512

UNKNOWN = '<UNKNOWN>'

random.seed(seed)

# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

start_time = time.time()


def elapsed():
    sec = time.time() - start_time
    if sec < 60:
        return "%.2f sec" % sec
    elif sec < 60 * 60:
        return "%.2f min" % (sec / 60.0)
    else:
        return "%.2f hour" % (sec / (60 * 60))


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
    mask = expanduser('~/testdata.clean/deploy/PDF32000_2008.txt')
    file_list = sorted(glob(mask))
    print('%d files' % len(file_list))
    return '\n'.join(read_text(path) for path in file_list)


RE_SPACE = re.compile(r'[\s,\.:;!\?\-\(\)]+', re.DOTALL | re.MULTILINE)
RE_NUMBER = re.compile(r'[^A-Z^a-z]')


def tokenize_simple(text, max_words):
    words = RE_SPACE.split(text)
    words = [w for w in words if len(w) > 1 and not RE_NUMBER.search(w)]
    if max_words > 0:
        words = words[:max_words]
    return [words]


spacy_nlp = spacy.load('en')
punct = {',', '.', ';', ':', '\n', '\t', '\f'}


def tokenize_spacy(text, n_input, max_words):
    document = spacy_nlp(text)
    doc_sents = list(document.sents)
    print(len(doc_sents))
    # for i, sent in enumerate(doc_sents[:20]):
    #     print(i, type(sent), len(sent), sent[:5])
    # sentences
    sentences = []
    n_words = 0
    for span in document.sents:
        sent = [token.text for token in span]
        sent = [w for w in sent if w not in punct]
        if len(sent) < n_input + 1:
            continue
        sentences.append(sent)
        n_words += len(sent)
        if max_words > 0 and n_words >= max_words:
            break
        # for token in sentence:
        #     print(token)
        #     token_dict = {}
        #     token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
        #     token_dict['text'] = text[token_dict['start']:token_dict['end']]
        #     if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
        #         continue
        #     # Make sure that the token text does not contain any space
        #     if len(token_dict['text'].split(' ')) != 1:
        #         print("WARNING: the text of the token contains space character, replaced with "
        #               "hyphen\n\t{0}\n\t{1}".format(token_dict['text'], token_dict['text'].replace(' ', '-')))
        #         token_dict['text'] = token_dict['text'].replace(' ', '-')
        #     sentence_tokens.append(token_dict)
        # sentences.append(sentence_tokens)
    print('!!!', len(sentences))
    for i, sent in enumerate(sentences[:5]):
        print(i, type(sent), len(sent), sent[:5])
    return sentences


def tokenize(text, n_input, max_words):
    if use_spacy:
        return tokenize_spacy(text, n_input, max_words)
    return tokenize_simple(text, max_words)


def build_indexes(sentences, max_vocab):
    word_counts = defaultdict(int)
    for sent in sentences:
        for w in sent:
            word_counts[w] += 1
    vocabulary_list = sorted(word_counts, key=lambda x: (-word_counts[x], x))[:max_vocab - 1]
    vocabulary_list.append(UNKNOWN)
    vocabulary = set(vocabulary_list)
    # for i, w in enumerate(words[:5]):
    #     marker = '***' if w in vocabulary else ''
    #     print('%3d: %-20s %s' % (i, w, marker))

    vocab_size = len(vocabulary)
    unk_index = vocab_size - 1
    word_index = {w: i for i, w in enumerate(vocabulary_list)}
    word_index[UNKNOWN] = unk_index
    index_word = {i: w for w, i in word_index.items()}

    # for i in sorted(index_word)[:5]:
    #     print(i, index_word[i], type(i))

    # for i in range(vocab_size):
    #     assert i in index_word, i

    return word_index, index_word, unk_index


def build_embeddings(word_index):
    embeddings = {w: np.zeros(len(word_index), dtype=np.float32) for w in word_index}
    for w, i in word_index.items():
        embeddings[w][i] = 1.0
    unk_embedding = embeddings[UNKNOWN]
    return embeddings, unk_embedding


text = make_text()
print('%7d bytes' % len(text))
sentences = tokenize(text, n_input, MAX_WORDS)
print('%7d sentences' % len(sentences))
print('%7d words' % sum(len(sent) for sent in sentences))
word_index, index_word, unk_index = build_indexes(sentences, MAX_VOCAB)
vocab_size = len(word_index)
print('%7d vocab' % vocab_size)
embeddings, unk_embedding = build_embeddings(word_index)
n_samples = sum((len(sent) - n_input) for sent in sentences)
print('%7d samples' % n_samples)


def batch_getter(sentences, n_input, batch_size):
    """Generator that returns x, y, oneh_y in `batch_size` batches
        phrase is a random phrase of length n_input + 1 from words
        x = indexes of first n_input words
        y = index of last word
        returns x, y, one hot encoding of y
    """
    sequence_numbers = []
    for i, sent in enumerate(sentences):
        for j in range(len(sent) - n_input - 1):
            sequence_numbers.append((i, j))
    random.shuffle(sequence_numbers)

    for k0 in range(0, len(sequence_numbers), batch_size):
        n = min(len(sequence_numbers) - k0, batch_size)
        # print('****', k, n, len(sequence_numbers))
        oneh_y = np.empty((n, vocab_size), dtype=np.float32)
        indexes_x = np.empty((n, n_input), dtype=int)
        indexes_y = np.empty((n), dtype=int)

        for k in range(n):
            i, j = sequence_numbers[k0 + k]
            words = sentences[i]
            assert j < len(words) - n_input - 1, (i, j, len(words), n_input)
            assert j + n_input < len(words), 'i=%d j=%d words=%d sequence_numbers n_input=%d' % (
                i, j, len(words), len(sequence_numbers), n_input)

            phrase_x = words[j:j + n_input]
            phrase_y = words[j + n_input]
            wx = [word_index.get(w, unk_index) for w in phrase_x]
            wy = word_index.get(phrase_y, unk_index)

            indexes_x[k] = np.array(wx)
            indexes_y[k] = np.array(wy)
            oneh_y[k] = embeddings.get(phrase_y, unk_embedding)

        # show('indexes_x', indexes_y)
        yield indexes_x, indexes_y, oneh_y


# tf Graph inputs
# x = indexes of first n_input words in phrase
# y = one hot encoding of last word in phrase
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
biases = tf.Variable(tf.random_normal([vocab_size]))


def RNN(x, weights, biases):
    """RNN predicts y (one hot) from x (indexes), weights and biases
        y = g(x) * W + b, where
            g = LSTM
            x = indexes of input words
            y = one-hot encoding of output word

    """
    show('x', x)

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)
    show('x split', x)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but we only want the last output
    y = tf.matmul(outputs[-1], weights) + biases
    show('y', y)
    return y


pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

os.makedirs(model_folder, exist_ok=True)
saver = tf.train.Saver(max_to_keep=3)

# Launch the graph
with tf.Session() as session:
    session.run(init)
    writer.add_graph(session.graph)
    best_acc = 0.0
    best_epoch = -1

    if False:
        devices = session.list_devices()
        for d in devices:
            print(d.name)
        assert False

    for epoch in range(n_epochs):
        accuracies = []
        acc_total = 0.0
        loss_total = 0.0
        source = batch_getter(sentences, n_input, batch_size)

        # Process minibatches of size `batch_size`
        for step, (indexes_x, indexes_y, onehot_y) in enumerate(source):
            # print('*** %s %d %d' % (list(indexes_y.shape), n_samples, batch_size))
            frac = len(indexes_y) / n_samples

            # Update the model
            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred],
                                                    feed_dict={x: indexes_x,
                                                               y: onehot_y})
            loss_total += loss * frac
            acc_total += acc * frac
            accuracies.append(acc)
            assert acc <= 1.0, (acc, loss, frac, accuracies)
            assert frac <= 1.0, (acc, loss, frac, accuracies)
            assert acc_total <= 1.0, (acc, loss, frac, accuracies)
            # assert (len(accuracies) + 1) * batch_size > n_samples, (len(accuracies), batch_size, n_samples)

        # Show some progress statistics
        if acc_total > best_acc:
            best_epoch = epoch
            best_acc = acc_total
            saver.save(session, os.path.join(model_folder, 'model_%06d.ckpt' % step))
            print('epoch=%3d acc=%.2f' % (epoch + 1, 100.0 * acc_total))

        done = epoch > best_epoch + patience or epoch == n_epochs - 1
        if (epoch + 1) % n_epochs_report == 0 or done:
            print("epoch=%3d (best=%3d): Average Loss=%9.6f, Average Accuracy=%5.2f%%" %
                  (epoch + 1, best_epoch + 1, loss_total, 100.0 * acc_total))

            # indexes = [int(indexes_x[0, i]) for i in range(indexes_x.shape[1])]
            # symbols_in = [index_word.get(i, UNKNOWN) for i in indexes]
            # symbols_out = index_word.get(indexes_y[0], UNKNOWN)
            # v = tf.argmax(onehot_pred, 1).eval()
            # symbols_out_pred = index_word[int(v[0])]
            # print("%s -> [%s] predicted [%s]" % (symbols_in, symbols_out, symbols_out_pred))

        if done:
            break

    print("Optimization Finished!")
    print("Elapsed time: ", elapsed())
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")
    # while True:
    #     prompt = "%s words: " % n_input
    #     sentence = input(prompt)
    #     sentence = sentence.strip()
    #     words = sentence.split(' ')
    #     if len(words) != n_input:
    #         continue
    #     try:
    #         indexes = [word_index.get(w, unk_index) for w in words]
    #         for i in range(32):
    #             keys = np.reshape(np.array(indexes), [-1, n_input])
    #             onehot_pred = session.run(pred, feed_dict={x: keys})
    #             indexes_pred = int(tf.argmax(onehot_pred, 1).eval())
    #             sentence = "%s %s" % (sentence, index_word[indexes_pred])
    #             indexes = indexes[1:]
    #             indexes.append(indexes_pred)
    #         print(sentence)
    #     except KeyError:
    #         print("Word not in dictionary")
