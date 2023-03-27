from collections.abc import Iterable


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
    else:
        for k in list(d)[:n]:
            print(f'{k:10s}: {peek_head(d[k])}')
        print(f'... {len(d) - 2 * n} items ...')
        for k in list(d)[-n:]:
            print(f'{k:10s}: {peek_head(d[k])}')