import os
import shutil


def peek_head(e, n: int = 3):
    if type(e) != str:
        try:
            l = list(e)
            if len(l) < n:
                return f"[{', '.join(l)}]"
            else:
                return f"[{', '.join(l[:n])}, ... {len(l)} elements in total ...]"
        except Exception as err:
            pass
            # print(err)

    return e


def print_head_tail(d: dict, n: int = 3):
    if len(d) < n:
        for k, v in d.items():
            print(f'{k:10s}: {peek_head(v)}')
        return

    if n == 0:
        print(f'... {len(d)} items ...')
        return

    for k in list(d)[:n]:
        print(f'{k:10s}: {peek_head(d[k])}')
    print(f'... {len(d) - 2 * n} items ...')
    for k in list(d)[-n:]:
        print(f'{k:10s}: {peek_head(d[k])}')


def clear_dir(paths: list) -> bool:
    for path in paths:
        if not os.path.exists(path):
            print(f'path not exist, {path}')
            return False
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    return True


def get_clip_paths(dataset: str, partition: str, clip_id, sets: list = ['gt', 'lq'],
                   fmt: str = '{dataset}/{partition}/{set}/{clip_id:04}'):
    # e.g. data/REDS/test/lq/clip_id
    return [fmt.format(dataset=dataset, partition=partition, clip_id=clip_id, set=set) for set in sets]
