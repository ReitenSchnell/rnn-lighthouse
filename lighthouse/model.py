import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import time
import random
import numpy as np
import os

RNN_SIZE = 256
NUM_LAYERS = 2
GRAD_CLIP = 5.
NUM_EPOCHS = 50
LEARNING_RATE = 0.002
DECAY_RATE = 0.97
LOG_DIR = "logs"
SAVE_DIR = "save"
SAVE_EVERY = 50


def train(batch_size, seq_length, vocab_size, x_data, y_data):
    cell = setup_cell()
    input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
    targets = tf.placeholder(tf.int32, [batch_size, seq_length])
    initial_state = cell.zero_state(batch_size, tf.float32)
    batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
    inc_batch_pointer_op = tf.assign(batch_pointer, batch_pointer + 1)
    epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
    batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
    tf.summary.scalar("time_batch", batch_time)

    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))

    main_scope = 'light'
    with tf.variable_scope(main_scope):
        softmax_w = tf.get_variable("softmax_w", [RNN_SIZE, vocab_size])
        variable_summaries(softmax_w)
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        variable_summaries(softmax_b)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, RNN_SIZE])
            inputs = tf.split(tf.nn.embedding_lookup(embedding, input_data), seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, initial_state, cell, scope=main_scope)
    output = tf.reshape(tf.concat(outputs, 1), [-1, RNN_SIZE])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = legacy_seq2seq.sequence_loss_by_example([logits],
                                                   [tf.reshape(targets, [-1])],
                                                   [tf.ones([batch_size * seq_length])],
                                                   vocab_size)
    cost = tf.reduce_sum(loss) / batch_size / seq_length
    tf.summary.scalar("cost", cost)
    final_state = last_state
    lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), GRAD_CLIP)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR)
    num_batches = len(x_data)

    with tf.Session() as sess:
        train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        for epoch_number in range(epoch_pointer.eval(), NUM_EPOCHS):
            sess.run(tf.assign(lr, LEARNING_RATE * (DECAY_RATE ** epoch_number)))
            state = sess.run(initial_state)
            speed = 0
            assign_op = epoch_pointer.assign(epoch_number)
            sess.run(assign_op)
            for batch_number in range(0, num_batches):
                start = time.time()
                x, y = x_data[batch_number], y_data[batch_number]
                feed = {input_data: x, targets: y, initial_state: state, batch_time: speed}
                summary, train_loss, state, _, _ = sess.run([merged, cost, final_state, train_op, inc_batch_pointer_op],
                                                            feed)
                total_batches = epoch_number * num_batches + batch_number
                train_writer.add_summary(summary, total_batches)
                speed = time.time() - start
                if total_batches % batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                          .format(total_batches, NUM_EPOCHS * num_batches, epoch_number, train_loss, speed))

                last_batch = epoch_number == NUM_EPOCHS - 1 and batch_number == num_batches - 1
                if total_batches % SAVE_EVERY == 0 or last_batch:
                    checkpoint_path = os.path.join(SAVE_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=total_batches)
                    print("model saved to {}".format(checkpoint_path))
        train_writer.close()
        print('training is finished')


def setup_cell():
    cell_fn = rnn.BasicLSTMCell
    cells = []
    for _ in range(NUM_LAYERS):
        layer_cell = cell_fn(RNN_SIZE)
        cells.append(layer_cell)
    cell = rnn.MultiRNNCell(cells)
    return cell


def sample(vocab_inv, vocab, sample_length=30):
    with tf.Session() as sess:
        cell = setup_cell()
        input_data = tf.placeholder(tf.int32, [1, 1])
        initial_state = cell.zero_state(1, tf.float32)

        main_scope = 'light'
        vocab_size = len(vocab)
        with tf.variable_scope(main_scope, reuse=tf.AUTO_REUSE):
            softmax_w = tf.get_variable("softmax_w", [RNN_SIZE, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [vocab_size, RNN_SIZE])
                inputs = tf.split(tf.nn.embedding_lookup(embedding, input_data), 1, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, initial_state, cell, scope=main_scope)
        output = tf.reshape(tf.concat(outputs, 1), [-1, RNN_SIZE])
        logits = tf.matmul(output, softmax_w) + softmax_b
        probs = tf.nn.softmax(logits)
        final_state = last_state

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('starting sampling')
            state = sess.run(initial_state)
            prime = random.choice(list(vocab.keys()))
            for word in prime.split()[:-1]:
                print('prime is:' + word)
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {input_data: x, initial_state: state}
                [state] = sess.run([final_state], feed)
            ret = prime
            word = prime.split()[-1]
            for n in range(sample_length):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {input_data: x, initial_state: state}
                [state_probs, state] = sess.run([probs, final_state], feed)
                p = state_probs[0]
                t = np.cumsum(p)
                s = np.sum(p)
                sample = int(np.searchsorted(t, np.random.rand(1)*s))
                pred = vocab_inv[sample]
                ret += ' ' + pred
                word = pred
            print('sampling finished')
            print('sampling result: ' + ret)
