# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-11-20 17:38

"""
生成"纲"、"种"多分类的训练与测试数据标签。预测是哺乳纲还是鸟纲，兔子、老鼠还是鸡。
"""

import os
from PIL import Image
import pandas as pd

ROOTS = '../Dataset/'
PHASE = ['train', 'val']
# [0,1]
CLASSES = ['Mammals', 'Birds']
# [0,1,2]
SPECIES = ['rabbits', 'rats', 'chickens']

DATA_info = {'train': {'path': [], 'classes': [], 'species': []},
             'val': {'path': [], 'classes': [], 'species': []}
             }

for p in PHASE:
    for s in SPECIES:
        DATA_DIR = ROOTS + p + '/' + s
        DATA_NAME = os.listdir(DATA_DIR)
        for item in DATA_NAME:
            try:
                img = Image.open(os.path.join(DATA_DIR, item))
            except OSError:
                pass
            else:
                DATA_info[p]['path'].append(os.path.join(DATA_DIR, item))
                # 按"纲"分类
                if s == 'rabbits' or s == 'rats':
                    DATA_info[p]['classes'].append(0)
                else:
                    DATA_info[p]['classes'].append(1)
                # 按"种"分类
                if s == 'rabbits':
                    DATA_info[p]['species'].append(0)
                elif s == 'rats':
                    DATA_info[p]['species'].append(1)
                else:
                    DATA_info[p]['species'].append(2)
    ANNOTATION = pd.DataFrame(DATA_info[p])
    ANNOTATION.to_csv('Multi_%s_annotation.csv' % p)
    print('Multi_%s_annotation file is saved.' % p)
