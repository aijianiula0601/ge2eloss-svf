from modules import *
from params import pm
import hparams as hp


class SvfOjb:
    def __init__(self, is_training=False):
        self.utterances_per_speaker = pm.utterances_per_speaker
        self.hidden_units = hp.model_hidden_size
        self.embed_size = hp.model_embedding_size
        self.n_mels = hp.mel_n_channels
        self.num_layer = hp.model_num_layers
        self.gpu_num = len(pm.gpu_devices.split(','))

        assert pm.batch_size % self.gpu_num == 0, 'expect batch_size / gpu_num is zero,but get batch_size:{} gpu_num:{}'.format(
            pm.batch_size, self.gpu_num)
        self.speakers_per_batch = int(pm.batch_size / self.gpu_num)
        self.batch_size = pm.batch_size

        if is_training:
            # shape=[batch_size,utterances_per_speaker,n_frames,n_mels]
            self.inpt = tf.placeholder(tf.float32,
                                       shape=[pm.batch_size, pm.utterances_per_speaker, None, hp.mel_n_channels])

            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                               trainable=False)

            assert pm.lr_decay_type in ["transformer",
                                        "exponential",
                                        "constant",
                                        "half"], 'lr_decay_type must in ["transformer", "exponential","constant"],but here is {}'.format(
                pm.lr_decay_type)

            self.lr = learning_rate_scheme(init_lr=pm.learning_rate,
                                           warmup_steps=pm.warmup_steps,
                                           global_step=self.global_step,
                                           min_learn_rate=pm.min_learn_rate,
                                           decay_rate=pm.decay_rate,
                                           lr_decay_type=pm.lr_decay_type)

            tf.summary.scalar('lr', self.lr)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

    def build_graph(self, inputs, is_training=False):
        """
        :param inputs:Tensor,which shape is [speakers_per_batch,utterances_per_speaker,n_frames_per_audio,n_mels]
        :param is_training:Bool,True for training and False for inference
        """

        # reshape the input to shape=[speaker_per_batch*utterances_per_speaker,n_frames,n_mels]
        inputs = tf.reshape(inputs, shape=[self.speakers_per_batch * self.utterances_per_speaker, -1, self.n_mels])

        # shape=[speaker_per_batch*utterances_per_speaker,embeded_size]
        embeds = embedding_block(inputs=inputs,
                                 hidden_units=self.hidden_units,
                                 embed_size=self.embed_size,
                                 num_layer=self.num_layer,
                                 training=True)

        # 测试阶段，先固定attention的参数
        # embeds = embedding_multi_heads_attention_block(inputs=inputs,
        #                                                rnn_hidden_units=self.hidden_units,
        #                                                attention_hidden_units=256,
        #                                                num_heads=4,
        #                                                embed_size=self.embed_size,
        #                                                rnn_num_layer=self.num_layer,
        #                                                attention_num_layer=4,
        #                                                drop_prob=0.3,
        #                                                attention_drop_prob=0.3,
        #                                                training=True)

        # shape=[speakers_per_batch,utterances_per_speaker,embed_size]
        embeds = tf.reshape(embeds, shape=[self.speakers_per_batch, self.utterances_per_speaker, self.embed_size])

        if is_training:
            self.loss, self.target, self.sim_matrix = calculate_loss(embeds)

    def build_inference(self):
        """
        build the inference graph,get speakers' embeddings
        :return: embeds,[Tensor],shape=[speakers_per_batch,utterances_per_speaker,embed_size]
        """
        with tf.variable_scope('gpu_model', reuse=tf.AUTO_REUSE):
            self.inpt_inference = tf.placeholder(tf.float32, shape=[None, None, self.n_mels])

            # shape=[bath_size,embeded_size]
            self.embeds = embedding_block(inputs=self.inpt_inference,
                                          hidden_units=self.hidden_units,
                                          embed_size=self.embed_size,
                                          num_layer=self.num_layer,
                                          training=False)

    def build_model(self):
        """
        :param gpu_num: how many gpu to use
        """
        assert self.batch_size % self.gpu_num == 0
        batch_size_slice = int(self.batch_size / self.gpu_num)

        loss_cl = []
        acc_cl = []
        tower_gradients = []
        for i in range(self.gpu_num):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('model_tower_{}'.format(i)):
                    with tf.variable_scope('gpu_model', reuse=i > 0):
                        with tf.name_scope('food'):
                            gpu_inputs = self.inpt[i * batch_size_slice:(i + 1) * batch_size_slice]

                        with tf.name_scope("model"):
                            self.build_graph(gpu_inputs, True)

                        tf.get_variable_scope().reuse_variables()

                        # combine loss
                        loss_cl.append(self.loss)
                        # Compute gradients for model parameters using tower's mini-batch
                        gradients = self.optimizer.compute_gradients(self.loss)
                        # clip_by_global_norm gradients avoid exploding gradient, keep the gradient directions
                        gradients = clip_by_global_norm(gradients)
                        # Retain tower's gradients
                        tower_gradients.append(gradients)

        # 计算所有GPU的loss,acc平均
        self.avg_loss = tf.reduce_mean(loss_cl)
        tf.summary.scalar('loss', self.avg_loss)

        loss_cl.clear()
        acc_cl.clear()

        # 平均所有的梯度
        grads = average_gradients(tower_gradients)

        # Add histograms for grads.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Track the moving averages of all trainable variables.
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            variable_averages = tf.train.ExponentialMovingAverage(0.9999, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        self.train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Build the summary operation from the last tower summaries.
        self.summary_op = tf.summary.merge_all()
