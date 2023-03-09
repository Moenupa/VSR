import glob, os
import tensorflow as tf

def _parse_function(record):
    features = {
        'id': tf.io.VarLenFeature(tf.string),
        'label': tf.io.VarLenFeature([1], tf.int64)
    }
    sample = tf.parse_single_example(record, features)
    return sample

if __name__ == '__main__':
    root_dir = os.path.join("data", "../data/video")
    path = glob.glob(f"{root_dir}/*.tfrecord")
    path.sort()
    print(path)

    raw_ds = tf.data.TFRecordDataset(path)
    for raw_record in raw_ds.take(1):
        eg = tf.train.Example()
        eg.ParseFromString(raw_record.numpy())
        print(eg)

    parsed_ds = raw_ds.map(_parse_function)

    for r in parsed_ds.take(1):
        print(r)

    exit(0)

    n = parsed_ds.as_numpy_iterator().next()
    length = len()
    print(f'len: {length}')
    output_id = []
    output_labels = []
    with tf.Session() as sess:
        for _ in range(length):
            id = sess.run(n)
            output_id.append(id)
        print(output_id)
            # output_labels