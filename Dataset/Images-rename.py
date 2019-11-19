import os


class ImageRename(object):
    """
    将数据集中乱七八糟的图片名批量重命名
    """
    def __init__(self, a, b):
        self.path = a + '/' + b

    def rename(self):
        file_list = os.listdir(self.path)
        total_num = len(file_list)
        n = 0
        for item in file_list:
            if item.endswith('.jpg'):
                old_name = os.path.join(os.path.abspath(self.path), item)
                new_name = os.path.join(os.path.abspath(self.path), s + format(str(n), '0>3s') + '.jpg')
                os.rename(old_name, new_name)
                n += 1
        print('total %d to rename & converted %d jpgs' % (total_num, n))


PHASE = ['train', 'val']
SPECIES = ['rabbits', 'rats', 'chickens']

for p in PHASE:
    for s in SPECIES:
        name = ImageRename(p, s)
        name.rename()
