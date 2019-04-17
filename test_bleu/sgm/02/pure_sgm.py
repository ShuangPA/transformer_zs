#!/usr/bin/python
#coding: utf-8

'''
将sgm中间的<hl> headline </hl>都给去掉。这个也许会影响概率训练。
'''

import re
from sys import argv

def stats(fe):
    reg = re.compile("<doc.*?</doc>",  re.IGNORECASE | re.DOTALL)
    reg0 = re.compile("<seg.*?</seg>", re.IGNORECASE | re.DOTALL)
    txt = file(fe, "rU").read()
    print len(txt)
    for docid, docs in enumerate(reg.findall(file(fe, "rU").read())):
        print docid, len(reg0.findall(docs)), "have <hl>" if "<hl>" in docs else "doesn't have <hl>" 

if __name__ == "__main__":
    for fe in argv[1:]:
        print fe
        stats(fe)



