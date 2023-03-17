from urllib3 import PoolManager


class YT8M_Translator():
    HOME = 'https://storage.googleapis.com/data.yt8m.org'
    # HOME = 'http://data.yt8m.org'
    CATEGORY_LOOKUP = f'{HOME}/2/j/v'
    VIDEO_LOOKUP = f'{HOME}/2/j/i'

    def __init__(self, num_pools: int = 4):
        self.manager = PoolManager(num_pools=num_pools)

    def translate(self, url: str) -> tuple:
        response = self.manager.request('GET', url)
        return eval(response.data[1:-1])

    def translate_vid(self, _id: str):
        return self.translate(
            f'{YT8M_Translator.VIDEO_LOOKUP}/{_id[:2]}/{_id}.js')

    def translate_cat(self, _id: str):
        return self.translate(f'{YT8M_Translator.CATEGORY_LOOKUP}/{_id}.js')


if __name__ == '__main__':
    translater = YT8M_Translator()
    code, ls = translater.translate_cat('03h_4m')
    print(code, ls[0])
