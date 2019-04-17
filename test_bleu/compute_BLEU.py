#!/usr/bin/env python3
import sys
sys.path.append('/home/shuangzhao/insight_nlp')
import pa_nlp.common as nlp

def compute(input_file):
  nlp.execute_cmd(f"./test_bleu/newextract.py -s test_bleu/sgm/05/nist_c2e_05_src "
                  f"-i {input_file} -o {input_file}.sgm")
  nlp.execute_cmd(f"./test_bleu/post_process -o {input_file}.sgm -n {input_file}.sgm.post -l")
  nlp.execute_cmd(f"./test_bleu/mteval-v11b.pl -c -r test_bleu/sgm/05/nist_c2e_05_ref -s test_bleu/sgm/05/nist_c2e_05_src "
                  f"-t {input_file}.sgm.post  > {input_file}.sgm.post.bleu.case")
  nlp.execute_cmd(f"./test_bleu/mteval-v11b.pl -r test_bleu/sgm/05/nist_c2e_05_ref -s test_bleu/sgm/05/nist_c2e_05_src "
                  f"-t {input_file}.sgm.post  > {input_file}.sgm.post.bleu.uncase")
  f= open(f"{input_file}.sgm.post.bleu.uncase", 'r').readlines()
  out = 0
  for line in f:
    if 'BLEU score =' in line:
      out = line.split(' ')[-4]
      break
  nlp.execute_cmd(f"rm {input_file}.sgm*")
  return out


