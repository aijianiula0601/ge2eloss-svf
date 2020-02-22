# -*- coding:utf-8 -*-
import multiprocessing

# 图形审核接口
bind = '0.0.0.0:5112'
backlog = 20
workers = 1
workers_class = 'gevent'
threads = 60
timeout = 60