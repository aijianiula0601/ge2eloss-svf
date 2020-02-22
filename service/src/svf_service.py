# -*- coding: utf-8 -*-
import json
import tensorflow as tf
from timeit import default_timer as timer
from flask import Flask, request, abort, jsonify

from service.config.logger import MyLogger
from service.model.model import SvfModel

logger = MyLogger.__call__().get_logger()

# 限定使用的gpu和使用gpu的内存
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.005
# sess = tf.Session(config=config)
# set_session(sess)


sess = tf.Session()

app = Flask(__name__)

model = None
graph = None


# 初始化模型
def load_model():
    global model

    model = SvfModel(sess)
    model.init_model()

    global graph
    graph = tf.get_default_graph()


# 加载模型
load_model()


@app.route('/api/1.0/asr', methods=['POST'])
def check():
    request_id = request.args.get('requestId', None)

    request_data = json.loads(request.data.decode('utf-8'))

    if request_id is None:
        return jsonify({'code': 10001, 'message': "not requestId!"})

    if 'wav_bodys' not in request_data:
        return jsonify({'requestId': str(request_id),
                        'code': 10002,
                        'message': "not wav_bodys!"
                        })

    wav_bodys = request_data['wav_bodys']
    try:
        # 获取特征
        process_start = timer()
        embed_result = model.get_audios_embeds(sess, request_id, wav_bodys)
        get_feature_time = timer() - process_start

        result_json = {
            'code': 1100,
            'message': 'Success!',
            'requestId': str(request_id),
            'send_num': len(wav_bodys),
            'return_num': len(embed_result),
            'avg_time(ms)': round(get_feature_time * 1000 / len(embed_result), 2)
        }
        logger.info("response:" + json.dumps(result_json))

        result_json['embeds'] = embed_result

        return jsonify(result_json)

    except Exception as e:
        result_json = {'requestId': str(request_id),
                       'code': 1,
                       'message': str(e)
                       }
        logger.error('response:' + json.dumps(result_json))
        return jsonify(result_json)


if __name__ == '__main__':
    """
    音频文件转文本
    访问URL: url = 'http://localhost:5112/api/1.0/asr'
    """
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5112)
