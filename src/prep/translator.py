from urllib3 import PoolManager
from time import sleep
import random
import pandas as pd
import json
import os
from config import Config
import logging
import shutil


def peek_head(item, n: int = 3):
    if type(item) == str:
        return item
    elif type(item) == list:
        if len(item) < n:
            return f"{', '.join(item)}"
        else:
            return f"{', '.join(item[:n])}, ... {len(item) - n} more ..."
    return item


def print_head_tail(d: dict, n: int = 3):
    if len(d) < n:
        for k, v in d.items():
            print(f'{k:10s}: {peek_head(v)}')
    else:
        for k in list(d)[:n]:
            print(f'{k:10s}: {peek_head(d[k])}')
        print(f'... {len(d) - 2 * n} items ...')
        for k in list(d)[-n:]:
            print(f'{k:10s}: {peek_head(d[k])}')


class YT8M_Translator():
    HOME = 'https://storage.googleapis.com/data.yt8m.org'
    # HOME = 'http://data.yt8m.org'
    CATEGORY_LOOKUP = f'{HOME}/2/j/v'
    VIDEO_LOOKUP = f'{HOME}/2/j/i'

    def __init__(self, num_pools: int = 4):
        self.backup_file = 'meta/translation.json'

        self.manager = PoolManager(num_pools=num_pools)

        self.translation = {'vid': {}, 'cat': {}}
        self.restore()

    def restore(self):
        logging.info('restore from last checkpoint')

        if os.path.exists(self.backup_file):
            self.translation = json.load(open(self.backup_file, 'r'))

    def backup(self):
        logging.info('backup to checkpoint')
        if os.path.exists(self.backup_file):
            shutil.copy2(self.backup_file, f'{self.backup_file}.bak')
        json.dump(self.translation, open(self.backup_file, 'w'))

    def update_vid_from_category(self):
        vid_set = set()
        for ls in self.translation['cat'].values():
            vid_set.update(ls)

        for key in vid_set:
            if self.translation['vid'].get(key) is None:
                self.translation['vid'][key] = ""
        self.backup()

    def translate(self, url: str, timeout: float = 0.05) -> tuple:
        count = 0
        while count < 4:
            response = self.manager.request('GET', url)
            sleep(timeout)
            if response.status != 200:
                count += 1
                continue

            if len(response.data) == 0:
                break
            return eval(response.data[1:-1])
        return None, None

    def translate_vid(self, _id: str):
        '''get youtube id from fake id'''
        if not self.translation['vid'].get(_id):
            fake_vid, ytb_vid = self.translate(
                f'{YT8M_Translator.VIDEO_LOOKUP}/{_id[:2]}/{_id}.js')
            if ytb_vid is None:
                logging.warning(f'{_id} -> None')
                return
            self.translation['vid'][fake_vid] = ytb_vid
            logging.info(f'{_id} -> {ytb_vid}')

    def translate_cat(self, _id: str):
        '''get a list of video fake-id from category code'''
        if not self.translation['cat'].get(_id):
            cat_id, fake_vid_list = self.translate(
                f'{YT8M_Translator.CATEGORY_LOOKUP}/{_id}.js')
            if fake_vid_list:
                logging.info(f'{_id} -> {peek_head(fake_vid_list)}')
                self.translation['cat'][cat_id] = fake_vid_list

    def parse_categories(self,
                         filepath: str = 'meta/yt8m_categories.csv',
                         n: int = 100):
        try:
            pd.read_csv(filepath,
                        header=None).iloc[:n, 0].apply(self.translate_cat)
        except Exception as e:
            print(e)
        finally:
            self.backup()

    def parse_videos(self, n: int = 2000):
        count = 0
        try:
            while count < n:
                key, val = random.choice(list(self.translation['vid'].items()))
                if val:
                    continue
                self.translate_vid(key)
                count += 1
        except Exception as e:
            logging.error(e)
        finally:
            self.backup()

    def peek(self):
        print_head_tail(self.translation['cat'])
        print('-' * 20)
        print_head_tail(self.translation['vid'], n=2)
        print(
            f"{'total':10s}: {sum(1 for v in self.translation['vid'].values() if v)}"
        )

    def parse_from_log(self, log: str):
        with open(log, 'r') as f:
            while line := f.readline():
                if '->' in line:
                    key, val = line.split(' -> ')
                    key, val = key.strip()[-4:], val.strip()
                    if key in self.translation['vid'] and val != 'None':
                        self.translation['vid'][key] = val
        self.backup()

    def get_vid(self, _id: str):
        return self.translation['vid'].get(_id)

    def get_cat(self, _id: str):
        return self.translation['cat'].get(_id)

    def get_all_vid(self):
        return list(v for v in self.translation['vid'].values() if v)


if __name__ == '__main__':
    config = Config(stdout=True, dry_run=False)
    translater = YT8M_Translator()
    translater.parse_categories()
    translater.update_vid_from_category()
    translater.parse_videos()
    translater.peek()
