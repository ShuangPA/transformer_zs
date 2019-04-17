# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import sys
sys.path.append('/home/shuangzhao/insight_nlp')
import pa_nlp.common as nlp
import tensorflow as tf
import time
from model import Transformer
#from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu, cal_oov
import os
from hparams import Hparams
import math
import logging
from test_bleu.compute_BLEU import compute

logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
hp.logdir += hp.id
hp.evaldir += hp.id

save_hparams(hp, hp.logdir)

print('-'*80)
cal_oov(hp.train1, hp.train2, hp.ch_vocab, hp.en_vocab)
cal_oov(hp.eval1, hp.eval2, hp.ch_vocab, hp.en_vocab)
print('-'*80)

os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu
logging.info(f"# using GPU:{hp.gpu}")
logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.en_vocab, hp.ch_vocab, hp.batch_size,
                                             shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2,
                                             100000, 100000,
                                             hp.en_vocab, hp.ch_vocab, hp.batch_size,
                                             shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries, lr = m.train(xs, ys)
y_hat, eval_summaries = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=1000)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
        _id = 0
    else:
        saver.restore(sess, ckpt)
        _id = int(str(ckpt).split('/')[-1].split('_')[1][2:])

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)

    cur_t = 0
    for i in range(_gs, total_steps+1):
        s_t = time.time()
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)
        _loss = sess.run(loss)
        print(f"step:{i}---loss:{_loss}")
        cur_t += (time.time() - s_t)

        if _gs and cur_t >= hp.eval_every_n_second:
            _id += 1
            cur_t = 0
            logging.info("evaluate id {} at epoch {}".format(_id, epoch))
            _loss = sess.run(loss) # train loss
            _lr = sess.run(lr)

            logging.info("# test evaluation")
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.en_idx2token)

            logging.info("# write results")
            model_output = f"NMT_ID{_id:05}_E{epoch:05}_L{round(float(_loss), 3)}_lr{round(float(_lr), 6)}"
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# calc bleu score and append it to result")
            #calc_bleu(hp.eval3, translation)
            final_bleu = compute(translation)
            result_path = os.path.join(hp.evaldir, "evaluate.out")
            _f = open(result_path, 'a')
            _f.write(f"{model_output.replace('_','  ')}  BLEU:{final_bleu} \n")
            _f.close()



            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()


logging.info("Done")
