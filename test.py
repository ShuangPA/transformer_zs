# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Inference
'''

import os
import time
import tensorflow as tf

from data_load import get_batch
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, postprocess, load_hparams
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
hp.testdir += hp.id
load_hparams(hp, hp.ckpt)

os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu
logging.info(f"# using GPU:{hp.gpu}")

logging.info("# Prepare test batches")
test_batches, num_test_batches, num_test_samples  = get_batch(hp.test1, hp.test1,
                                              100000, 100000,
                                              hp.en_vocab, hp.ch_vocab, hp.test_batch_size,
                                              shuffle=False)
print("-"*80)
print(f"number of test batches: {num_test_batches}")
print(f"number of test samples: {num_test_samples}")
print("-"*80)

iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
xs, ys = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
m = Transformer(hp)
y_hat, _ = m.eval(xs, ys)

logging.info("# Session")
saver = tf.train.Saver()
with tf.Session() as sess:
    #ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    #ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
    ckpt = hp.ckpt

    #saver = tf.train.import_meta_graph('/home/shuangzhao/transformer_1/mt_log/g213_2/NMT_E6_L2.444_lr7e-05_-73488.meta')

    saver.restore(sess, ckpt)

    sess.run(test_init_op)

    logging.info("# get hypotheses")
    t1 = time.time()
    hypotheses = get_hypotheses(num_test_batches, num_test_samples, sess, y_hat, m.en_idx2token)
    t2 = time.time()
    print("-" * 80)
    print(f"Time for getting results: {t2 - t1}")

    logging.info("# write results")
    model_output = ckpt.split("/")[-1]
    if not os.path.exists(hp.testdir): os.makedirs(hp.testdir)
    translation = os.path.join(hp.testdir, model_output)
    with open(translation, 'w') as fout:
        fout.write("\n".join(hypotheses))

    logging.info("# calc bleu score and append it to translation")
    #calc_bleu(hp.test2, translation)
    print(f"Time rest: {time.time() - t2}")
