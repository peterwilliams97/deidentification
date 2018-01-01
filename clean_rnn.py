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
use_glove_embeddings = True
embedding_size = 100
embeddings_freeze = True
embeddings_path = expanduser('~/data/glove.6B/glove.6B.100d.txt')
INDEX_ALL_WORDS = True
MAX_VOCAB = 50000
MAX_WORDS = 80000
SMALL_TEXT = False
HOLMES = True
if SMALL_TEXT:
    HOLMES = False
learning_rate = 0.001
n_steps = 3
batch_size = 100
test_frac = .2
n_epochs = 1000
n_epochs_report = 1
patience = 100

model_folder = 'models'

# number of units in RNN cell
hidden_size = 512
UNKNOWN = '<UNKNOWN>'
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

start_time = time.time()


def elapsed():
    sec = time.time() - start_time
    if sec < 60:
        return "%.2f sec" % sec
    elif sec < 60 * 60:
        return "%.2f min" % (sec / 60.0)
    else:
        return "%.2f hour" % (sec / (60 * 60))


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
    return
    if isinstance(x, (list, tuple)):
        v = '%d:%s' % (len(x), describe(x))
    else:
        v = describe(x)
    print('"%s"=%s' % (name, v))


def read_text(path):
    with open(path, 'rt') as f:
        return f.read()


def make_text():
    if SMALL_TEXT:
        return read_text('belling_the_cat.txt')
    if HOLMES:
        return read_text('cano_bare.txt')
    # mask = expanduser('~/testdata.clean/deploy/PDF32000_2008.txt')
    # mask = expanduser('~/testdata.clean/deploy/The_Block_Cipher_Companion.txt')
    mask = expanduser('~/testdata.clean/deploy/Folk_o_bostadsrakningen_1970_12.txt')

    file_list = sorted(glob(mask))
    print('%d files' % len(file_list))
    return '\n'.join(read_text(path) for path in file_list)


RE_SPACE = re.compile(r'[\s,\.:;!\?\-\(\)]+', re.DOTALL | re.MULTILINE)
RE_NUMBER = re.compile(r'[^A-Z^a-z]')

spacy_nlp = spacy.load('en')
punct = {',', '.', ';', ':', '\n', '\t', '\f', ''}


def tokenize(text, n_steps, max_words):
    document = spacy_nlp(text)
    doc_sents = list(document.sents)
    print(len(doc_sents))
    sentences = []
    n_words = 0
    for span in document.sents:
        sent = [token.text.strip() for token in span]
        sent = [w for w in sent if w not in punct]
        if len(sent) < n_steps + 1:
            continue
        sentences.append(sent)
        n_words += len(sent)
        if max_words > 0 and n_words >= max_words:
            break

    print('!!!', len(sentences))
    for i, sent in enumerate(sentences[:5]):
        print(i, type(sent), len(sent), sent[:5])
    return sentences


def train_test(sentences, n_steps, test_frac):
    """Spilt sentences list into training and test lists.
        training will have test_frac of the words in sentences
    """
    random.shuffle(sentences)
    n_samples = sum((len(sent) - n_steps) for sent in sentences)
    n_test = int(n_samples * test_frac)
    test = []
    i_test = 0
    for sent in sentences:
        if i_test + len(sent) - n_steps > n_test:
            break
        test.append(sent)
        i_test += len(sent) - n_steps
    train = sentences[len(test):]
    assert train and test, (len(sentences), len(test), len(train), test_frac)
    return train, test


def build_indexes(sentences, max_vocab):
    word_counts = defaultdict(int)
    for sent in sentences:
        for w in sent:
            word_counts[w] += 1
    vocabulary_list = sorted(word_counts, key=lambda x: (-word_counts[x], x))[:max_vocab - 1]
    vocabulary_list = [UNKNOWN] + vocabulary_list
    # vocabulary = set(vocabulary_list)
    # # for i, w in enumerate(words[:5]):
    # #     marker = '***' if w in vocabulary else ''
    # #     print('%3d: %-20s %s' % (i, w, marker))

    unk_index = 0
    word_index = {w: i for i, w in enumerate(vocabulary_list)}
    index_word = {i: w for w, i in word_index.items()}

    # for i in sorted(index_word)[:5]:
    #     print(i, index_word[i], type(i))

    # for i in range(vocabulary_size):
    #     assert i in index_word, i

    return word_counts, word_index, index_word, unk_index


def load_word_embeddings(path):
    import codecs
    count = -1
    word_vector = {}
    with codecs.open(path, 'r', 'UTF-8') as f:
        for line in f:
            count += 1
            # if count > 1000:break
            line = line.strip()
            line = line.split(' ')
            if not line:
                continue
            token = line[0]
            vector = np.array([float(x) for x in line[1:]])
            word_vector[token] = vector

    sizes = {v.size for v in word_vector.values()}
    assert len(sizes) == 1, (len(sizes), sorted(sizes)[:10])
    return word_vector


def build_embeddings(word_index, embeddings_path):
    word_vector = load_word_embeddings(embeddings_path)
    v = next(iter(word_vector.values()))
    unk_embedding = np.zeros(v.shape, dtype=v.dtype)
    embeddings = {w: word_vector.get(w, unk_embedding) for w in word_index}
    del word_vector
    return embeddings, unk_embedding


#
# Execution starts here
#
text = make_text()
print('%7d bytes' % len(text))
sentences = tokenize(text, n_steps, MAX_WORDS)
print('ALL')
print('%7d sentences' % len(sentences))
print('%7d words' % sum(len(sent) for sent in sentences))

if INDEX_ALL_WORDS:
    word_counts, word_index, index_word, unk_index = build_indexes(sentences, MAX_VOCAB)

train, test_sentences = train_test(sentences, n_steps, test_frac)
sentences = train

print('TRAIN')
print('%7d sentences' % len(sentences))
print('%7d words' % sum(len(sent) for sent in sentences))

if not INDEX_ALL_WORDS:
    word_counts, word_index, index_word, unk_index = build_indexes(sentences, MAX_VOCAB)

vocabulary_size = len(word_index)
print('%7d vocab' % vocabulary_size)
print('%7d embedding_size' % embedding_size)

# not used !@#$
# embeddings, unk_embedding = build_embeddings(word_index, embeddings_path)
n_samples = sum((len(sent) - n_steps) for sent in sentences)
print('%7d samples' % n_samples)

print('hidden_size=%d' % hidden_size)
print('vocabulary_size=%d' % vocabulary_size)
print('n_steps=%d' % n_steps)
print('batch_size=%d' % batch_size)


def batch_getter(sentences, n_steps, batch_size):
    """Generator that returns x, y, oneh_y in `batch_size` batches
        phrase is a random phrase of length n_steps + 1 from words
        x = indexes of first n_steps words
        y = index of last word
        returns x, y
    """
    sequence_numbers = []
    for i, sent in enumerate(sentences):
        for j in range(len(sent) - n_steps - 1):
            sequence_numbers.append((i, j))
    random.shuffle(sequence_numbers)

    for k0 in range(0, len(sequence_numbers), batch_size):
        n = min(len(sequence_numbers) - k0, batch_size)
        # print('****', k, n, len(sequence_numbers))
        indexes_x = np.empty((n, n_steps), dtype=int)
        indexes_y = np.empty(n, dtype=int)

        for k in range(n):
            i, j = sequence_numbers[k0 + k]
            words = sentences[i]
            assert j < len(words) - n_steps - 1, (i, j, len(words), n_steps)
            assert j + n_steps < len(words), 'i=%d j=%d words=%d sequence_numbers n_steps=%d' % (
                i, j, len(words), len(sequence_numbers), n_steps)

            phrase_x = words[j:j + n_steps]
            phrase_y = words[j + n_steps]
            wx = [word_index.get(w, unk_index) for w in phrase_x]
            wy = word_index.get(phrase_y, unk_index)

            indexes_x[k] = np.array(wx)
            indexes_y[k] = np.array(wy)

        # show('indexes_x', indexes_y)
        yield indexes_x, indexes_y


def RNN(x, weights, biases):
    """RNN predicts y (one hot) from x (indexes), weights and biases
        y = g(x) * W + b, where
            g = LSTM
            x = indexes of input words
            y = one-hot encoding of output word
    """
    show('x in', x)

    # 1-layer LSTM with hidden_size units.
    rnn_cell = rnn.BasicLSTMCell(hidden_size)

    # generate prediction
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    outputs0 = outputs[:, -1, :]

    tf.summary.histogram('sum_outputs', outputs)

    # there are n_steps outputs but we only want the last output
    logits = tf.nn.xw_plus_b(outputs0, weights, biases)
    show('outputs', outputs)
    show('outputs[-1]', outputs0)
    show('weights', weights)
    show('biases', biases)
    show('logits', logits)
    logits = tf.reshape(logits, [-1, 1, vocabulary_size])
    show('logits out', logits)
    return logits


initializer = tf.contrib.layers.xavier_initializer()

with tf.name_scope("inputs"):
    # tf Graph inputs
    # X = indexes of first `n_steps` words in phrase
    # y = index of last word in phrase
    X = tf.placeholder(tf.float32, [None, n_steps], name="X")
    y = tf.placeholder(tf.int64, [None], name="y")
    X1 = tf.expand_dims(X, -1, name='X1')
    y1 = tf.expand_dims(y, -1, name='y1')

show('X', X)
show('X1', X1)
show('y', y)
show('y1', y1)

with tf.name_scope("model"):
    # RNN output node weights and biases
    weights = tf.Variable(tf.random_normal([hidden_size, vocabulary_size]), name='weights')
    biases = tf.Variable(tf.random_normal([vocabulary_size]), name='biases')

    # # # https://www.tensorflow.org/programmers_guide/embedding
    # word_embeddings = tf.get_variable("word_embeddings", [vocabulary_size, embedding_size])
    # embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)

    # # embeddings = tf.Variable(
    # #         tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # # #     embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # token_embedding_weights = tf.get_variable(
    #                  "token_embedding_weights",
    #                   shape=[vocabulary_size, embedding_size],
    #                   initializer=initializer,
    #                   trainable=not embeddings_freeze)

    logits = RNN(X1, weights, biases)
    logits_shape = tf.shape(logits, name="logits_shape")
    y_shape = tf.shape(y1, name="y_shape")

    tf.summary.histogram('sum_biases', biases)
    tf.summary.histogram('sum_weights', weights)
    tf.summary.histogram('sum_logits', logits)

show("logits", logits)
show("logits_shape", logits_shape)
show("y", y)
show("y1", y1)
show("y_shape", y_shape)
print('batch_size=%d' % batch_size)

with tf.name_scope("loss"):
    # Loss and optimizer
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    # Use the contrib sequence loss and average over the batches
    # loss0 = tf.contrib.seq2seq.sequence_loss(
    #         logits,
    #         y1,
    #         tf.ones([logits_shape[0], 1], dtype=tf.float32),
    #         average_across_timesteps=False,  # There is only one output timestep
    #         average_across_batch=True)
    # cost0 = tf.reduce_mean(loss0)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y1, name="xentropy")
    cost = tf.reduce_mean(xentropy, name="cost")
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost, name="optimizer")

    # show("loss0", loss0)
    show("loss", xentropy)
    # show("cost0", cost0)
    show("cost", cost)
    # assert False

    top = tf.argmax(logits, 2, name="top")

    # Model evaluation !@#$ For embeddings?
    show("tf.argmax(logits, 2)", top)

    correct_pred0 = tf.equal(top, y1, name="correct_pred0")
    correct_pred = tf.cast(correct_pred0, tf.float32, name="correct_pred")
    accuracy = tf.reduce_mean(correct_pred, name="accuracy")

    tf.summary.histogram('sum_xentropy', xentropy)
    tf.summary.scalar('sum_cost', cost)
    tf.summary.histogram('sum_correct_pred', correct_pred)
    tf.summary.scalar('sum_accuracy', accuracy)


def demonstrate(indexes_x, indexes_y, logits, step, i):
    indexes = [int(indexes_x[i, j]) for j in range(indexes_x.shape[1])]
    symbols_in = [index_word.get(j, UNKNOWN) for j in indexes]
    symbols_out = index_word.get(int(indexes_y[i]), UNKNOWN)
    v = tf.argmax(logits, 2).eval()
    show('indexes_x', indexes_x)
    show('indexes_y', indexes_y)
    show('logits', logits)
    show('v', v)
    symbols_out_pred = index_word.get(int(v[i, 0]), UNKNOWN)
    # print('symbols_out=%s' % symbols_out)
    # print('symbols_out_pred=%s' % symbols_out_pred)
    mark = '****' if symbols_out_pred == symbols_out else ''
    return "%5d: %60s -> [%s] -- [%s] %s" % (step, symbols_in, symbols_out, symbols_out_pred, mark)


def make_prediction(session, x):
    return tf.run([y], feed_dict={X: x})


def test_model(session, x):
    return tf.run([accuracy], feed_dict={X: x})


def test_results(session):
    batch_size = 1000

    acc_total = 0.0
    loss_total = 0.0
    predictions = []

    source = batch_getter(test_sentences, n_steps, batch_size)

    step = 0
    # Process minibatches of size `batch_size`
    for _, (indexes_x, indexes_y) in enumerate(source):
        # print('*** %s %d %d' % (list(indexes_y.shape), n_samples, batch_size))
        frac = len(indexes_y) / n_samples

        # Update the model
        acc, loss, y_pred = session.run([accuracy, cost, logits],
                                feed_dict={X: indexes_x,
                                           y: indexes_y})
        loss_total += loss * frac
        acc_total += acc * frac
        assert acc <= 1.0, (acc, loss, frac)
        assert frac <= 1.0, (acc, loss, frac)
        assert acc_total <= 1.0, (acc, loss)

        for i in range(len(indexes_y)):
            if step < 10:
                predictions.append(demonstrate(indexes_x, indexes_y, y_pred, step, i))
            step += 1

    return loss_total, acc_total, predictions


# Initializing the variables
init = tf.global_variables_initializer()

merged_summary = tf.summary.merge_all()
# Target log path
logs_path = './tf_rnn_words'
train_writer = tf.summary.FileWriter(logs_path)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
saver = tf.train.Saver(max_to_keep=1)
os.makedirs(model_folder, exist_ok=True)

# Launch the graph
print('@@0')
with tf.Session() as session:
    print('@@1')
    session.run(init)
    print('@@2')
    train_writer.add_graph(session.graph)
    print('@@3')

    # load_embeddings_from_pretrained_model(session, word_index, index_word)
    print('@@4')

    best_acc = 0.0
    best_epoch = -1
    new_best = False

    if False:
        devices = session.list_devices()
        for d in devices:
            print(d.name)
        assert False

    for epoch in range(n_epochs):
        accuracies = []
        train_acc = 0.0
        train_loss = 0.0
        source = batch_getter(sentences, n_steps, batch_size)

        print('size 0: %d' % session.graph_def.ByteSize())

        # Process minibatches of size `batch_size`
        for step, (indexes_x, indexes_y) in enumerate(source):
            # print('*** %s %d %d' % (list(indexes_y.shape), n_samples, batch_size))
            frac = len(indexes_y) / n_samples

            show('X', X)
            show('y', y)
            show('indexes_x', indexes_x)
            show('indexes_y', indexes_y)

            # Update the model
            _, acc, loss, y_pred = session.run([optimizer, accuracy, cost, logits],
                                               feed_dict={X: indexes_x,
                                                          y: indexes_y},
                                               options=run_options,
                                               run_metadata=run_metadata)
            train_loss += loss * frac
            train_acc += acc * frac
            accuracies.append(acc)
            assert acc <= 1.0, (acc, loss, frac)
            assert frac <= 1.0, (acc, loss, frac)
            assert train_acc <= 1.0, (acc, loss, frac)
            assert loss >= 0.0, (acc, loss, frac)
        test_loss, test_acc, predictions = test_results(session)
        # Show some progress statistics
        if test_acc > best_acc:
            best_epoch = epoch
            best_acc = test_acc
            best_predictions = predictions
            new_best = True
            print('epoch=%3d train_acc=%.2f test_acc=%.2f' % (epoch + 1,
                100.0 * train_acc, 100.0 * test_acc))
            run_number = 'run_number_%03d' % best_epoch
            saver.save(session, os.path.join(model_folder, run_number))
            # summary = tf.summary.FileWriter(logdir=os.path.join(logs_path, run_number), graph=session.graph)
            # train_writer.add_run_metadata(run_metadata, run_number)

        print('size 1: %d' % session.graph_def.ByteSize())
        summary = session.run(merged_summary,
                              feed_dict={X: indexes_x,
                                         y: indexes_y},
                              options=run_options,
                              run_metadata=run_metadata)
        print('size 2: %d' % session.graph_def.ByteSize())
        train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
        print('size 3: %d' % session.graph_def.ByteSize())
        train_writer.add_summary(summary, epoch)
        print('size 4: %d' % session.graph_def.ByteSize())
        # print('summary=%s' % summary)

        done = epoch > best_epoch + patience or epoch == n_epochs - 1
        if (epoch + 1) % n_epochs_report == 0 or done:
            print("epoch=%3d (best=%3d): Train Loss=%9.6f Accuracy=%5.2f%% -- "
                  "Test Loss=%9.6f  Accuracy=%5.2f%%" %
                  (epoch + 1, best_epoch + 1,
                   train_loss, 100.0 * train_acc,
                   test_loss, 100.0 * test_acc), flush=True)
            # run_metadata = tf.RunMetadata()
            # train_writer.add_run_metadata(run_metadata, 'step%d' % epoch)
            # train_writer.add_summary(summary, epoch)
            if new_best:
                print('\n'.join(best_predictions))
                new_best = False

            # indexes = [int(indexes_x[0, i]) for i in range(indexes_x.shape[1])]
            # symbols_in = [index_word.get(i, UNKNOWN) for i in indexes]
            # symbols_out = index_word.get(indexes_y[0], UNKNOWN)
            # v = tf.argmax(y_pred, 1).eval()
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
    #     prompt = "%s words: " % n_steps
    #     sentence = input(prompt)
    #     sentence = sentence.strip()
    #     words = sentence.split(' ')
    #     if len(words) != n_steps:
    #         continue
    #     try:
    #         indexes = [word_index.get(w, unk_index) for w in words]
    #         for i in range(32):
    #             keys = np.reshape(np.array(indexes), [-1, n_steps])
    #             y_pred = session.run(pred, feed_dict={x: keys})
    #             indexes_pred = int(tf.argmax(y_pred, 1).eval())
    #             sentence = "%s %s" % (sentence, index_word[indexes_pred])
    #             indexes = indexes[1:]
    #             indexes.append(indexes_pred)
    #         print(sentence)
    #     except KeyError:
    #         print("Word not in dictionary")
