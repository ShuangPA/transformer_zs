from utils import cal_oov


ch_file = '../Desktop/3w-unseg-mt-data/ch.eval.nist'
en_file = '../Desktop/3w-unseg-mt-data/en.eval.nist'
ch_vocab = '../Desktop/3w-unseg-mt-data/bpe_ch.vocab'
en_vocab = '../Desktop/3w-unseg-mt-data/unseg_en.vocab'
cal_oov(ch_file, en_file, ch_vocab, en_vocab)