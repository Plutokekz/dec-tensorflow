import tensorflow as tf
from dec.dataset import *
import os
import configargparse
from dec.model import *
import csv


def export_z(z, filename, metadata=None, metafilename=None):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        wz = csv.writer(f, delimiter='\t')
        for z_i in z:
            wz.writerow([z_i_j for z_i_j in z_i])

    if metafilename is not None:
        with open(metafilename, 'w', encoding='utf-8', newline='') as f:
            wm = csv.writer(f, delimiter='\t')
            for label_idx in metadata:
                wm.writerow([label_idx])


def inference(dataset,
              dec_ckpt_path,
              encoder_dims=[500, 500, 2000, 10],
              plot_filename='cluster.png'):

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if dataset == 'MNIST':
        data = MNIST()
    else:
        assert False, "Undefined dataset."

    model = DEC(params={
        "encoder_dims": encoder_dims,
        "n_clusters": data.num_classes,
        "input_dim": data.feature_dim,
        "alpha": 1.0,
    })

    checkpoint = tf.train.Checkpoint(model=model)
    restore_path = dec_ckpt_path
    if os.path.isdir(dec_ckpt_path):
        restore_path = tf.train.latest_checkpoint(dec_ckpt_path)
    checkpoint.restore(restore_path)

    z = model.ae.encode(data.train_x.astype('float32'), keep_prob=1.0).numpy()

    export_z(z, 'z.tsv', data.train_y, 'meta.tsv')
    return z


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("--gpu-index", dest="gpu_index", help="GPU Index Number", default="0", type=str)

    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_index']

    inference(dataset="MNIST",
              dec_ckpt_path="./dec_ckpt",
              plot_filename='cluster.png')
