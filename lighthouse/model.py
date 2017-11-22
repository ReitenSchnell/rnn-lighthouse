import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import time

RNN_SIZE = 256
NUM_LAYERS = 2
GRAD_CLIP = 5.
NUM_EPOCHS = 50
LEARNING_RATE = 0.002
DECAY_RATE = 0.97


def train(batch_size, seq_length, vocab_size, x_data, y_data):
    cell_fn = rnn.BasicRNNCell
    cells = []
    for _ in range(NUM_LAYERS):
        layer_cell = cell_fn(RNN_SIZE)
        cells.append(layer_cell)
    cell = rnn.MultiRNNCell(cells)
    input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
    targets = tf.placeholder(tf.int32, [batch_size, seq_length])
    initial_state = cell.zero_state(batch_size, tf.float32)
    batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
    inc_batch_pointer_op = tf.assign(batch_pointer, batch_pointer + 1)
    epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
    batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
    tf.summary.scalar("time_batch", batch_time)

    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [RNN_SIZE, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, RNN_SIZE])
            inputs = tf.split(tf.nn.embedding_lookup(embedding, input_data), seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, initial_state, cell, scope='rnnlm')
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
    num_batches = len(x_data)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for e in range(epoch_pointer.eval(), NUM_EPOCHS):
            sess.run(tf.assign(lr, LEARNING_RATE * (DECAY_RATE ** e)))
            state = sess.run(initial_state)
            speed = 0
            assign_op = epoch_pointer.assign(e)
            sess.run(assign_op)
            for b in range(0, num_batches):
                start = time.time()
                x, y = x_data[b], y_data[b]
                feed = {input_data: x, targets: y, initial_state: state, batch_time: speed}
                summary, train_loss, state, _, _ = sess.run([merged, cost, final_state, train_op, inc_batch_pointer_op],
                                                            feed)
                speed = time.time() - start
                if (e * num_batches + b) % batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                          .format(e * num_batches + b, NUM_EPOCHS * num_batches, e, train_loss, speed))

    print('training is finished')
