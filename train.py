# -*- coding: utf-8 -*-
import os
import sys
import logging
from keras.backend.tensorflow_backend import set_session

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

from model import *
from dataset.data_load import *


def init_dataset(meta_files):
    """
    初始化dataset
    :param meta_files:[list],The list contains the meta file
    """
    train_dataset, train_num_batch, dev_dataset, dev_num_batch = get_dataset(meta_files)

    train_iterator = train_dataset.make_initializable_iterator()
    train_next_element = train_iterator.get_next()

    dev_iterator = dev_dataset.make_initializable_iterator()
    dev_next_element = dev_iterator.get_next()

    return train_next_element, train_num_batch, train_iterator, dev_next_element, dev_num_batch, dev_iterator


if __name__ == '__main__':

    if not 'win' in sys.platform:
        import setproctitle

        setproctitle.setproctitle(pm.process_name)

    # 限定使用的gpu和使用gpu的内存
    os.environ["CUDA_VISIBLE_DEVICES"] = pm.gpu_devices
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=pm.log_device_placement)
    tf_config.gpu_options.per_process_gpu_memory_fraction = pm.gpu_memory_rate

    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
            # 创建多GPU的图
            start_time = time.time()
            svfObj = SvfOjb(is_training=True)

            svfObj.build_model()
            logging.info("Build model done! Duration time:{}s".format(time.time() - start_time))

            # 创建session
            sess = tf.Session(config=tf_config)
            set_session(session=sess)

            # 获取训练，验证数据集
            logging.info("Get dataset....")
            start_time = time.time()

            train_next_element, train_num_batch, train_iterator, dev_next_element, dev_num_batch, dev_iterator = init_dataset(
                pm.train_meta_files.split(","))

            # 模型保存
            train_summary_writer = tf.summary.FileWriter(pm.log_dir + os.sep + 'train', sess.graph)

            # 保存dev的模型
            dev_model_save_dir = pm.checkpoint + os.sep + 'dev'
            Path(dev_model_save_dir).mkdir(exist_ok=True)
            dev_summary_writer = tf.summary.FileWriter(pm.log_dir + os.sep + 'dev', sess.graph)
            # 保存效果最好的dev模型
            best_dev_model_save_dir = pm.checkpoint + os.sep + 'min_dev'
            Path(best_dev_model_save_dir).mkdir(exist_ok=True)

            # 保存模型
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=pm.max_keep_model)
            try:
                model_file = tf.train.latest_checkpoint(best_dev_model_save_dir)
                saver.restore(sess, model_file)
                logging.info('已加载最近训练模型!')
            except Exception as e:
                logging.info("无法加载旧模型,训练过程会覆盖旧模型!")
                pass

            if not os.path.exists(pm.checkpoint):
                os.mkdir(pm.checkpoint)

            sess.run(tf.global_variables_initializer())
            sess.run(train_iterator.initializer)
            sess.run(dev_iterator.initializer)

            train_loss_cl = []
            lr_cl = []
            min_dev_eer = 1e8
            for epoch in range(1, pm.epochs + 1):

                # 训练
                loss_cl = []
                for step in range(train_num_batch):
                    feature = sess.run(train_next_element)

                    sim_matrix, loss, st, lr, _, summary = sess.run([svfObj.sim_matrix,
                                                                     svfObj.loss,
                                                                     svfObj.global_step,
                                                                     svfObj.lr,
                                                                     svfObj.train_op,
                                                                     svfObj.summary_op],
                                                                    feed_dict={svfObj.inpt: feature})
                    loss_cl.append(loss)
                    logging.info(
                        'train--step:{}({}) global_step:{} loss:{}'.format(step, train_num_batch, int(st), loss))

                train_loss_cl.append(sum(loss_cl) / len(loss_cl))
                lr_cl.append(lr)
                eer = calculate_eer(sim_matrix)
                train_summary_writer.add_summary(summary, epoch)
                logging.info('------------------------------------------------------------------------------')
                logging.info("train epoch:{} gs:{}".format(epoch, int(st)))
                logging.info("lr:{}".format(lr))
                logging.info("avg_loss:{}".format(sum(loss_cl) / len(loss_cl)))
                logging.info("last batch eer:{}".format(eer))
                logging.info('------------------------------------------------------------------------------')
                logging.info("------------------------------------------------------------------------------")
                if pm.display_loss:
                    for ls, lr in zip(train_loss_cl, lr_cl):
                        logging.info("lr:{} loss:{}".format(lr, ls))
                    logging.info("------------------------------------------------------------------------------")
                loss_cl.clear()
                if epoch > 0 and epoch % 1 == 0:
                    # 验证集
                    eer_cl = []
                    for step in range(dev_num_batch):
                        feature = sess.run(dev_next_element)

                        sim_matrix, loss, st, lr, summary = sess.run([svfObj.sim_matrix,
                                                                      svfObj.loss,
                                                                      svfObj.global_step,
                                                                      svfObj.lr,
                                                                      svfObj.summary_op],
                                                                     feed_dict={svfObj.inpt: feature})

                        eer = calculate_eer(sim_matrix)
                        loss_cl.append(loss)
                        eer_cl.append(eer)
                        logging.info(
                            'dev--step:{}({}) global_step:{} loss:{} eer:{}'.format(step, dev_num_batch, int(st), loss,
                                                                                    eer))
                    dev_summary_writer.add_summary(summary, epoch)
                    logging.info('------------------------------------------------------------------------------')
                    logging.info("dev epoch:{} gs:{}".format(epoch, int(st)))
                    logging.info("lr:{}".format(lr))
                    logging.info("avg_loss:{}".format(sum(loss_cl) / len(loss_cl)))
                    logging.info("avg_eer:{}".format(sum(eer_cl) / len(eer_cl)))
                    logging.info('------------------------------------------------------------------------------')
                    cur_eer = np.mean(np.array(eer_cl))
                    saver.save(sess,
                               dev_model_save_dir + "/dev_mode_epoch_{}_gs_{}_eer_{:.6f}".format(epoch, st, cur_eer))
                    if cur_eer < min_dev_eer:
                        saver.save(sess,
                                   best_dev_model_save_dir + "/dev_mode_epoch_{}_gs_min_{}_eer_{:.6f}".format(epoch, st,
                                                                                                              cur_eer))
                        min_dev_eer = cur_eer

                    loss_cl.clear()
