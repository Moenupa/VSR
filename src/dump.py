import os
import pickle
import os.path as osp


def pickle_dump(obj, path):
    dirname = osp.dirname(path)
    if not osp.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    pickle.dump(obj, open(path, 'wb'))


if __name__ == '__main__':
    pickle_dump('test', 'data/STM/test.pkl')