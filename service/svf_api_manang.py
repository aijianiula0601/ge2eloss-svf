# -*- coding:utf-8 -*-

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)

print(project_dir)

from prjs.wangyi.svf.ge2e.service.src.svf_service import app

if 'win' not in sys.platform:
    import setproctitle

    setproctitle.setproctitle("asr_service")

if __name__ == '__main__':
    """
    音频转文字
    """
    app.run(host='0.0.0.0', port=5112)
