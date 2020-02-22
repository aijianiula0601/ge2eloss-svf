import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq


def similarity_matrix(embeds):
    """
    Computes the similarity matrix according the section 2.1 of GE2E.
    :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
    utterances_per_speaker, embedding_size)
    :return: the similarity matrix as a tensor of shape (speakers_per_batch,
    utterances_per_speaker, speakers_per_batch)
    """
    with tf.name_scope(name='similarity_matrix'):
        # Cosine similarity scaling (with fixed initial parameter values)

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
        sim_matrix = tf.concat(
            [tf.concat([tf.reduce_sum(centroids_excl[i] * embeds[j], axis=1,
                                      keep_dims=True) if i == j
                        else tf.reduce_sum(centroids_incl[i] * embeds[j], axis=1, keep_dims=True) for i
                        in range(speakers_per_batch)],
                       axis=1) for j in range(speakers_per_batch)], axis=0)

        return sim_matrix


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
