import tensorflow as tf
from dec.dataset import *
import os
import configargparse
from dec.model import *
import numpy as np


def train(dataset,
          batch_size=256,
          encoder_dims=[500, 500, 2000, 10],
          initialize_iteration=50000,
          finetune_iteration=100000,
          pretrained_ae_ckpt_path=None):

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

    log_interval = 5000

    # phase 1: parameter initialization
    if pretrained_ae_ckpt_path is None:
        sae = StackedAutoEncoder(encoder_dims=encoder_dims, input_dim=data.feature_dim)
        os.makedirs('ae_ckpt', exist_ok=True)

        next_ = data.gen_next_batch(batch_size=batch_size, is_train_set=True, iteration=initialize_iteration)
        cur_ae_data = data.train_x.astype(np.float32)
        for i, sub_ae in enumerate(sae.layerwise_autoencoders):
            for iter_, (batch_x, _, _) in enumerate(next_):
                loss = sub_ae.train_step(batch_x.astype(np.float32), keep_prob=0.8)
                if iter_ % log_interval == 0:
                    print("[SAE-{}] iter: {}\tloss: {}".format(i, iter_, float(loss)))

            # assign pretrained sub_ae weights to corresponding layers in model
            model.ae.dense_layers[i].assign_weights(sub_ae.dense_layers[0])
            model.ae.dense_layers[(i + 1) * -1].assign_weights(sub_ae.dense_layers[1])

            # get next sub_ae's input
            cur_ae_data = sub_ae.encode(cur_ae_data, keep_prob=1.0).numpy()
            embedding = Dataset(train_x=cur_ae_data, train_y=cur_ae_data)
            next_ = embedding.gen_next_batch(batch_size=batch_size, is_train_set=True, iteration=initialize_iteration)

        # finetune full AE
        for iter_, (batch_x, _, _) in enumerate(data.gen_next_batch(
                batch_size=batch_size, is_train_set=True, iteration=finetune_iteration)):
            loss = model.ae.train_step(batch_x.astype(np.float32), keep_prob=1.0)
            if iter_ % log_interval == 0:
                print("[AE-finetune] iter: {}\tloss: {}".format(iter_, float(loss)))

        checkpoint = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(checkpoint, 'ae_ckpt', max_to_keep=1)
        manager.save()
        ae_restore_path = manager.latest_checkpoint
    else:
        ae_restore_path = pretrained_ae_ckpt_path
        if os.path.isdir(ae_restore_path):
            ae_restore_path = tf.train.latest_checkpoint(ae_restore_path)

    # phase 2: parameter optimization
    os.makedirs('dec_ckpt', exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ae_restore_path)

    # initialize cluster centers via k-means
    z = model.ae.encode(data.train_x.astype(np.float32), keep_prob=1.0).numpy()
    model.get_assign_cluster_centers_op(z)

    dec_checkpoint = tf.train.Checkpoint(model=model)
    dec_manager = tf.train.CheckpointManager(dec_checkpoint, 'dec_ckpt', max_to_keep=None)

    for cur_epoch in range(50):
        q = model.soft_assignment(data.train_x.astype(np.float32)).numpy()
        p = model.target_distribution(q)

        for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(
                batch_size=batch_size, is_train_set=True, epoch=1)):
            batch_p = p[batch_idxs].astype(np.float32)
            loss, pred = model.train_step(batch_x.astype(np.float32), batch_p, keep_prob=0.8)

        print("[DEC] epoch: {}\tloss: {}\tacc: {}".format(
            cur_epoch, float(loss), model.cluster_acc(batch_y, pred.numpy())))
        dec_manager.save()


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("--batch-size", dest="batch_size", help="Train Batch Size", default=300, type=int)
    parser.add("--gpu-index", dest="gpu_index", help="GPU Index Number", default="0", type=str)

    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_index']

    train(batch_size=args['batch_size'],
          dataset="MNIST",
          pretrained_ae_ckpt_path="./ae_ckpt")
