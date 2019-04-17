#!/usr/bin/env python2

import sys
import os
import cmath
import time
import optparse
import re

reg_docid = re.compile('''docid="(.*?)"''', re.IGNORECASE)

def get_doc_id(ln):
    rst = reg_docid.findall(ln)
    assert rst != []
    return rst[0]

def extract(ln):
    p1 = ln.find(">") + 1
    p2 = ln.rfind("<")
    return [ln[: p1], ln[p1: p2], ln[p2: ]]

def analyze_tran(nbest):
    fin = file(nbest, "rU")
    n = 1082
    trans = []
    for i in xrange(n):
        m = 1
        assert(m > 0)
        tmp = []
        for j in xrange(m):
            line = fin.readline().rstrip()
            line = line.replace("<unk>", " ")
            tmp.append(line.split(" ||| ")[0])
        yield tmp[0]            
           
if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-s", dest = "src", help = "source")
    optparser.add_option("-i", dest = "infs", help = "input")
    optparser.add_option("-o", dest = "outfs", help = "output")
    (opts, args) = optparser.parse_args() 

    print "src:", opts.src
    print "nbest:", opts.infs
    print "out:", opts.outfs

    assert opts.src is not None
    assert opts.infs is not None
    assert opts.outfs is not None

    fin = file(opts.src, "rU")
    fou = file(opts.outfs, "w")
    
    dochead = fin.next().rstrip().replace("srcset", "tstset")
    print >> fou, dochead
    tran_ite = analyze_tran(opts.infs)

    num = 0
    while True:
        try:
            ln = fin.next().strip()
        except StopIteration:
            break
        ln = ln.rstrip().replace("</srcset>", "</tstset>")
        if "<doc" in ln or "<DOC" in ln or "<Doc" in ln:
            docname = get_doc_id(ln) 
            print >> fou, '''<doc docid="%s" sysid="chiero">''' %(docname)
        elif ln.startswith("<seg"):
            if "</seg>" in ln:
                h, b, t = extract(ln)
                print >> fou, h, tran_ite.next(), t 
            else:
                h = ln
                n = int(fin.next())
                for k in range(n):
                    fin.next()
                t = fin.next()
                print >> fou, h, tran_ite.next(), t
            num += 1
        else:
            print >> fou, ln
    fou.close()            
    print "OK"            
