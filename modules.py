import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from attenion_block import multi_head_attention, get_position_encoding


def embedding_block(inputs, hidden_units, embed_size, num_layer, drop_prob=0.4, training=False, scope='lstm'):
    """
    :param inputs: tensor,shape=[speakers_per_batch*utterances_per_speaker,n_frames_per_audio,n_mels]
    :param hidden_units: int,the lstm cell hidden units
    :param embed_size: int,the speaker d_vector
    :param num_layer: int,the number layers for rnn
    :param drop_prob: float,the probability to set the hidden units to zero
    :param training: bool,true for train,false for inference
    :param scope: (optional),the scope's name the this block
    :return Tensor,shape=[speakers_per_batch*utterances_per_speaker,1,embed_size]
    """

    with tf.name_scope(name='embedding_block'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            cells = [tf.contrib.rnn.LSTMCell(num_units=hidden_units) for i in range(num_layer)]
            if training:
                cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - drop_prob) for cell in cells]
            multi_cells = tf.contrib.rnn.MultiRNNCell(cells)
            outputs, states = tf.nn.dynamic_rnn(cell=multi_cells,
                                                inputs=inputs,
                                                dtype=tf.float32,
                                                time_major=False)

            # inputs is the final hidden state
            embeds = tf.layers.dense(inputs=states[-1].h, units=embed_size, activation=tf.nn.relu)

            embeds = tf.nn.l2_normalize(embeds, axis=1)

        return embeds


def embedding_multi_heads_attention_block(inputs,
                                          rnn_hidden_units,
                                          attention_hidden_units,
                                          num_heads,
                                          embed_size,
                                          rnn_num_layer,
                                          attention_num_layer,
                                          drop_prob=0.4,
                                          attention_drop_prob=0.4,
                                          training=False,
                                          scope='attention_embedding'):
    """
    :param inputs: tensor,shape=[speakers_per_batch*utterances_per_speaker,n_frames_per_audio,n_mels]
    :param rnn_hidden_units: int,the lstm cell hidden units
    :param attention_hidden_units: int,the attention cell hidden units
    :param num_heads: int,the number of heads for attention
    :param embed_size: int,The output embedding vector dimension
    :param rnn_num_layer: int,The number of layer for rnn
    :param attention_num_layer: int,The number of layers for attention layer
    :param drop_prob: float,the probability to set the hidden units to zero in rnn layers
    :param attention_drop_prob: float,the probability to set the hidden units to zero in attention layers
    :param training: bool,true for train,false for inference
    :param scope: (optional),the scope's name the this block
    :return:
    """
    # inputs add position encode
    length = tf.shape(inputs)[1]
    pos_encoding = get_position_encoding(length, tf.shape(inputs)[-1])
    encoder_inputs = inputs + pos_encoding

    with tf.name_scope(name='embedding_multi_heads_attention_block'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # encoder
            for i in range(attention_num_layer):
                with tf.variable_scope('attention_layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                    encoder_inputs = multi_head_attention(x=encoder_inputs,
                                                          y=encoder_inputs,
                                                          hidden_size=attention_hidden_units,
                                                          num_heads=num_heads,
                                                          attention_dropout=attention_drop_prob,
                                                          training=training)

            # embedding
            embeds = embedding_block(inputs=encoder_inputs, hidden_units=rnn_hidden_units, embed_size=embed_size,
                                     num_layer=rnn_num_layer, drop_prob=drop_prob, training=training)

        return embeds


def _similarity_matrix(embeds):
    """
    Computes the similarity matrix according the section 2.1 of GE2E.
    :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
    utterances_per_speaker, embedding_size)
    :return: the similarity matrix as a tensor of shape (speakers_per_batch,
    utterances_per_speaker, speakers_per_batch)
    """
    with tf.name_scope(name='similarity_matrix'):
        # Cosine similarity scaling (with fixed initial parameter values)
        similarity_weight = tf.get_variable('similarity_weight', [], initializer=tf.constant_initializer(10.0))
        similarity_bias = tf.get_variable('similarity_bias', [], initializer=tf.constant_initializer(-5.0))

        speakers_per_batch, utterances_per_speaker = embeds.shape.as_list()[:2]
        # Inclusive centroids (1 per speaker),shape=shape=(speakers_per_batch,1, embedding_size)
        centroids_incl = tf.reduce_mean(embeds, axis=1, keep_dims=True)
        centroids_incl = tf.nn.l2_normalize(centroids_incl, axis=2)

        # Exclusive centroids (1 per utterance),shape=(speakers_per_batch,utterances_per_speaker, embedding_size)
        centroids_excl = tf.reduce_sum(embeds, axis=1, keep_dims=True) - embeds
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = tf.nn.l2_normalize(centroids_excl, axis=2)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.

        # method 1
        # temp_list = []
        # for speaker_j in range(speakers_per_batch):
        #     speaker_j_utterances_sim_speaker_i_list = []
        #     for speaker_i in range(speakers_per_batch):
        #         if speaker_i != speaker_j:
        #             center_vector = centroids_incl[speaker_i]
        #         else:
        #             center_vector = centroids_excl[speaker_i]
        #
        #         speaker_j_utterances_sim_to_speaker_i = tf.reduce_sum(embeds[speaker_j] * center_vector, axis=1,
        #                                                               keepdims=True)
        #         speaker_j_utterances_sim_speaker_i_list.append(speaker_j_utterances_sim_to_speaker_i)
        #     speaker_i_sim_matrix = tf.concat(speaker_j_utterances_sim_speaker_i_list, axis=1)
        #     temp_list.append(tf.expand_dims(speaker_i_sim_matrix, axis=0))
        # sim_matrix = tf.concat(temp_list, axis=0)

        # method 2, is equal to method 1
        sim_matrix = tf.concat(
            [tf.concat([tf.reduce_sum(centroids_excl[i] * embeds[j], axis=1,
                                      keep_dims=True) if i == j
                        else tf.reduce_sum(centroids_incl[i] * embeds[j], axis=1, keep_dims=True) for i
                        in range(speakers_per_batch)],
                       axis=1) for j in range(speakers_per_batch)], axis=0)

        sim_matrix = sim_matrix * similarity_weight + similarity_bias
        return sim_matrix


def calculate_loss(embeds):
    """
    Computes the softmax loss according the section 2.1 of GE2E.

    :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
    utterances_per_speaker, embedding_size)
    :return: Three values:
             loss:Tensor
             target(ground_truth):Tensor and shape=[speakers_per_batch * utterances_per_speaker]
             sim_matrix:Tensor and shape=[speakers_per_batch * utterances_per_speaker, speakers_per_batch])
    """
    speakers_per_batch, utterances_per_speaker = embeds.shape.as_list()[:2]

    # loss
    sim_matrix = _similarity_matrix(embeds)
    sim_matrix = tf.reshape(sim_matrix, shape=[speakers_per_batch * utterances_per_speaker, speakers_per_batch])
    ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
    target = tf.convert_to_tensor(ground_truth)
    # equal to :
    # target_one_hot = tf.one_hot(target, speakers_per_batch)
    # - tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(sim_matrix, axis=-1) * target_one_hot, axis=-1))
    loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=sim_matrix)

    return loss, target, sim_matrix


def calculate_eer(sim_matrix):
    """
    calculate the equal error rate.This function is executed with python not the tensorflow.
    :param sim_matrix:np.array.The similar table,shape=[speakers_per_batch * utterances_per_speaker, speakers_per_batch])
    :return:float
    """

    speakers_per_batch = np.shape(sim_matrix)[-1]
    assert np.shape(sim_matrix)[0] % speakers_per_batch == 0
    utterances_per_speaker = np.shape(sim_matrix)[0] // speakers_per_batch

    ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)

    # eer
    inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
    labels = np.array([inv_argmax(i) for i in ground_truth])

    # Snippet from https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = roc_curve(labels.flatten(), sim_matrix.flatten())
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return eer


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)

            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]

        if 'similarity_weight' in v.name or 'similarity_bias' in v.name:  # scale similarity weight and bias
            grad = 0.01 * grad

        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def clip_by_global_norm(gradients):
    gradients, v = zip(*gradients)
    gradients, _ = tf.clip_by_global_norm(gradients, 3.)

    return zip(gradients, v)


def learning_rate_scheme(init_lr, global_step, warmup_steps, decay_rate=0.85, min_learn_rate=0.00001,
                         lr_decay_type='transformer'):
    if lr_decay_type == 'transformer':
        learning_rate = init_lr * warmup_steps ** 0.5 * tf.minimum(global_step * warmup_steps ** -1.5,
                                                                   global_step ** -0.5)

    elif lr_decay_type == "exponential":
        decay_lr = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step, decay_steps=warmup_steps,
                                              decay_rate=decay_rate, staircase=False)

        learning_rate = tf.where(global_step <= warmup_steps, init_lr, decay_lr)

        learning_rate = tf.where(learning_rate <= min_learn_rate, min_learn_rate, learning_rate)

    elif lr_decay_type == "half":

        # step每过10000次，学习率减半
        decay_steps = 10000
        decay_half_rate = (global_step + 1) // decay_steps
        if_update = False
        if (global_step + 1) % decay_steps == 0 and decay_half_rate >= 1:
            if_update = True

        learning_rate = tf.where(if_update, init_lr, init_lr / (2 * decay_half_rate))

    else:
        learning_rate = tf.constant(init_lr, dtype=tf.float32)

    return learning_rate
