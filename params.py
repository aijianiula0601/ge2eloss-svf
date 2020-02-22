# -*- coding: utf-8 -*-
import argparse
import os
import multiprocessing

cur_dir = os.path.dirname(os.path.abspath(__file__))
print(cur_dir)


class Params:
    '''
    Parameters of our model
    '''

    def __init__(self):
        self.initialized = False

        self.__parser = argparse.ArgumentParser()

        # ----------------
        # train
        # ----------------
        self.__parser.add_argument("--epochs", default=500, type=int, help="how many epochs to train")
        self.__parser.add_argument("--batch_size", default=4, type=int, help="how many rows to train once")
        self.__parser.add_argument("--max_keep_model", default=20, type=int, help="how many model to save")
        self.__parser.add_argument("--run_dev", default=True, type=bool, help="if to run dev")
        self.__parser.add_argument("--save_interval", default=1, type=int,
                                   help="how many epochs save the model at a time")
        self.__parser.add_argument("--each_epoch_steps", default=10, type=int,
                                   help="how many steps to run for one epoch")
        self.__parser.add_argument("--num_threads", default=4, type=int, help="The number of threads for data prepare")
        self.__parser.add_argument("--display_loss", default=False, type=bool, help="if diplay the training loss")

        # ----------------
        # model parameter
        # ----------------
        self.__parser.add_argument("--utterances_per_speaker", default=5, type=int,
                                   help="the number of utterances per speaker")
        self.__parser.add_argument("--dropout_rate", default=0.2, type=float, help="number of layers to encoder")
        self.__parser.add_argument("--warmup_steps", default=2000, type=int,
                                   help="the warmup_steps not to reduce the learning rate")
        self.__parser.add_argument("--decay_rate", default=0.8, type=float,
                                   help="decay_rate to decay the learning rate when the decay type is exponential")
        self.__parser.add_argument("--learning_rate", default=0.001, type=float,
                                   help="the learning rate to optimize the weights")
        self.__parser.add_argument("--lr_decay_type", default='transformer', type=str, help="learning rate decay type")
        self.__parser.add_argument("--min_learn_rate", default=0.00002, type=float, help="the min learning rate")
        # ----------------
        # gpu control
        # ----------------
        self.__parser.add_argument("--buffer_size", default=3, type=int,
                                   help="the number of batch_size prepare in queue")
        self.__parser.add_argument("--process_name", default="tts", type=str, help="process_name")
        self.__parser.add_argument("--gpu_devices", default="0,1", type=str, help="which gpu to use")
        self.__parser.add_argument("--thread_num", default=multiprocessing.cpu_count(), type=int,
                                   help="the number of thread to prepare dataset")
        self.__parser.add_argument("--gpu_memory_rate", default=1.0, type=float,
                                   help="the probability of memory to use in earch gpu")
        self.__parser.add_argument("--log_device_placement", default=False, type=bool,
                                   help="if print out the logs")

        # ----------------
        # dataset
        # ----------------
        self.__parser.add_argument("--train_meta_files", default="dataset/test_data/meta.txt", type=str,
                                   help="files contains the training data.Separate by a comma")

        # ----------------
        # log
        # ----------------
        self.__parser.add_argument("--log_dir", default="/tmp/svf/log", type=str, help="the log directory")
        self.__parser.add_argument("--checkpoint", default="/tmp/svf/checkpoint", type=str,
                                   help="the directory to save model")

        # parse params
        self.__args = self.__parser.parse_args()

    def init_params(self):
        if self.initialized is True:
            return

        print('init_parms...')
        # -------------------
        # train
        # -------------------
        self.epochs = self.__args.epochs
        self.batch_size = self.__args.batch_size
        self.speakers_per_batch = self.__args.batch_size
        self.max_keep_model = self.__args.max_keep_model
        self.run_dev = self.__args.run_dev
        self.save_interval = self.__args.save_interval
        self.each_epoch_steps = self.__args.each_epoch_steps
        self.num_threads = self.__args.num_threads
        self.display_loss = self.__args.display_loss

        # -------------------
        # model parameter
        # -------------------
        self.utterances_per_speaker = self.__args.utterances_per_speaker
        self.dropout_rate = self.__args.dropout_rate
        self.warmup_steps = self.__args.warmup_steps
        self.decay_rate = self.__args.decay_rate
        self.learning_rate = self.__args.learning_rate
        self.lr_decay_type = self.__args.lr_decay_type
        self.min_learn_rate = self.__args.min_learn_rate

        # ----------------
        # gpu control
        # ----------------
        self.gpu_memory_rate = self.__args.gpu_memory_rate
        self.gpu_devices = self.__args.gpu_devices
        self.thread_num = self.__args.thread_num
        self.buffer_size = self.__args.buffer_size
        self.process_name = self.__args.process_name
        self.log_device_placement = self.__args.log_device_placement

        # ----------------
        # dataset
        # ----------------
        self.train_meta_files = self.__args.train_meta_files

        # -------------------
        # log
        # -------------------
        self.log_dir = self.__args.log_dir
        self.checkpoint = self.__args.checkpoint

        print("params:", str(self.__args.__dict__))


pm = Params()
pm.init_params()

if __name__ == '__main__':
    print(pm.batch_size)
