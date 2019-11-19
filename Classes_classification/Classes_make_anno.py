import pandas as pd
import os
from PIL import Image

"""
生成“纲”分类的训练与测试数据标签。
选兔子和鸡作为数据集，预测 属于哺乳纲(Mammals)还是鸟纲(Birds)。
"""

ROOTS = '../Dataset/'
PHASE = ['train', 'val']
CLASSES = ['Mammals', 'Birds']  # [0,1]
SPECIES = ['rabbits', 'chickens']

DATA_info = {'train': {'path': [], 'classes': []},
             'val': {'path': [], 'classes': []}
             }
for p in PHASE:
    for s in SPECIES:
        DATA_DIR = ROOTS + p + '\\' + s
        DATA_NAME = os.listdir(DATA_DIR)

        for item in DATA_NAME:
            try:
                img = Image.open(os.path.join(DATA_DIR, item))
            except OSError:
                pass
            else:
                DATA_info[p]['path'].append(os.path.join(DATA_DIR, item))
                if s == 'rabbits':
                    DATA_info[p]['classes'].append(0)
                else:
                    DATA_info[p]['classes'].append(1)

    ANNOTATION = pd.DataFrame(DATA_info[p])
    ANNOTATION.to_csv('Classes_%s_annotation.csv' % p)
    print('Classes_%s_annotation file is saved.' % p)
