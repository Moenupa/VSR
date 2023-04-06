import json
import logging
import math
import os
import random
import shutil
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from urllib3 import PoolManager

from config import Config
from utils import peek_head, print_head_tail


def plot_cat_distribution(n: int = 100, n_label: int = 100):
    translater = YT8M_Translator()
    fid_list = translater.get_translated_fakeid()
    vid_list = Dataset('data/STM').get_vid_list()

    df = translater.get_cat_distribution(fid_list, vid_list).iloc[:n, :]
    print(df)
    colors = 0.9 - np.random.rand(100, 3, ) / 2
    fig = plt.figure('STM category distribution', figsize=(2560/300, 1440/300), dpi=300, layout='tight')

    plt.scatter(
        df.iloc[:, 1],
        df.iloc[:, 2],
        s=df.iloc[:, 3],
        c=colors
    )
    plt.xlabel('YT8M Video Count Per Category')
    plt.ylabel('STM Translated Video Count Per Category')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e4, 1e6)
    plt.ylim(2e1, 1e4)
    for i in random.choices(np.arange(n), weights=df.iloc[:n, 1], k=round(math.log(n_label, 1.2))):
        name, x, y = df.iloc[i, :3].values
        noise = math.log(y + 1, 4e2)
        if i % 2 == 0:
            plt.text(x, y * noise, name, c=colors[i] / 2)
        else:
            plt.text(x, y / noise, name, c=colors[i] / 2)
    fig.savefig('STM_distribution_correlation')

    fig = plt.figure('YT8M categorical distribution', figsize=(2560/300, 1440/300), dpi=300, layout='tight')
    plt.barh(df.iloc[:, 0], df.iloc[:, 1], color=colors)
    plt.yticks(rotation=0, fontsize=4)
    plt.xscale('log')
    plt.margins(y=0)
    plt.ylabel('YT8M Tag Distribution')
    fig.savefig('YT8M_category_distribution')

    fig = plt.figure('STM categorical distribution', figsize=(2560 / 300, 1440 / 300), dpi=300, layout='tight')
    plt.barh(df.iloc[:, 0], df.iloc[:, 2], color=colors)
    plt.yticks(rotation=0, fontsize=4)
    plt.xscale('log')
    plt.margins(y=0)
    plt.ylabel('STM Tag Distribution')
    fig.savefig('STM_category_distribution')


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
        self.cat_dict = pd.read_csv('meta/yt8m_categories.csv', index_col=0)

    def restore(self):
        '''restore to a checkpoint file, translation.json'''
        logging.info('restore from last checkpoint')

        if os.path.exists(self.backup_file):
            self.translation = json.load(open(self.backup_file, 'r'))

    def backup(self):
        '''backup to a checkpoint file, translation.json'''
        logging.info('backup to checkpoint')
        if os.path.exists(self.backup_file):
            shutil.copy2(self.backup_file, f'{self.backup_file}.bak')
        json.dump(self.translation, open(self.backup_file, 'w'))

    def update_vid_from_category(self):
        '''add all video ids under `self.translation['cat']` to `self.translation['vid']` keys, default value is None'''
        vid_set = set()
        for ls in self.translation['cat'].values():
            vid_set.update(ls)

        for key in vid_set:
            if self.translation['vid'].get(key) is None:
                self.translation['vid'][key] = ""
        self.backup()

    def translate(self, url: str, timeout: float = 0.05) -> tuple:
        '''parse a YT8M url response to a tuple of (key, value)'''
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
        print_head_tail(self.translation['cat'], n=1)
        print('-' * 20)
        print_head_tail(self.translation['vid'], n=0)
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

    def lookup_vid(self, _id: str) -> str:
        '''get youtube id from fake id'''
        return self.translation['vid'].get(_id)

    def lookup_cat(self, _id: str) -> str:
        '''get category name from category id'''
        return self.cat_dict[self.cat_dict[0] == _id][1].values[0]

    def lookup_cat_num(self, _id: str) -> int:
        '''get category number from category id'''
        return self.cat_dict[self.cat_dict[0] == _id][2].values[0]

    def get_cat(self, _id: str) -> list:
        return self.translation['cat'].get(_id)

    def get_translated_fakeid(self) -> list:
        return list(k for k, v in self.translation['vid'].items() if v)

    def get_translated_vid(self, randomize: bool = True) -> list:
        ret = list(v for v in self.translation['vid'].values() if v)
        if randomize:
            random.shuffle(ret)
        return ret

    def get_cat_distribution(self, fid_list: list, vid_list: list) -> pd.DataFrame:
        translated_count_by_cat = {}
        downloaded_count_by_cat = {}
        for cat_name, d_fid_list in self.translation['cat'].items():
            translated_count_by_cat[cat_name] = len(set(fid_list).intersection(d_fid_list))
            d_vid_list = list(map(self.lookup_vid, d_fid_list))
            downloaded_count_by_cat[cat_name] = len(set(vid_list).intersection(d_vid_list))
        translated = pd.DataFrame.from_dict(translated_count_by_cat, orient='index', columns=['translated'])
        downloaded = pd.DataFrame.from_dict(downloaded_count_by_cat, orient='index', columns=['downloaded'])
        ret = pd.concat([self.cat_dict, translated, downloaded], axis=1).fillna(0).astype(dtype={'translated': int,
                                                                                                 'downloaded': int})
        return ret


if __name__ == '__main__':
    from dataset import Dataset
    config = Config(stdout=True, dry_run=False)
    # translater = YT8M_Translator()
    # translater.parse_categories()
    # translater.update_vid_from_category()
    # translater.parse_videos()
    # translater.peek()
    plot_cat_distribution()
