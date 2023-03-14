import tensorflow as tf
import glob
import json
from src.config import random_str, Config


def peek(raw_record):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)


def extract_fn(record):
    features = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.VarLenFeature(tf.int64)
    }
    return tf.io.parse_single_example(record, features)


def extract_to_dict(parsed_dataset) -> dict:
    ret = {}

    iterator = iter(parsed_dataset)
    record = iterator.get_next()

    try:
        while record:
            fake_vid = record['id'].numpy().decode()
            labels = record['labels'].values.numpy().tolist()
            ret[fake_vid] = labels
            record = iterator.get_next()
    except tf.errors.OutOfRangeError:
        pass

    return ret


def save_dict(d: dict, path: str) -> None:
    with open(path) as out:
        json.dump(d, out)


if __name__ == '__main__':
    config = Config(stdout=False, dry_run=False)
    raw_ds = tf.data.TFRecordDataset(glob.glob("data/video/*.tfrecord"))
    parsed_ds = raw_ds.map(extract_fn)

    video_label_pairs = extract_to_dict(parsed_ds)
    config.save_fp(lambda fp, obj: json.dump(obj, fp), 'labels.json', video_label_pairs)
