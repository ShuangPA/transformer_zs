lines = open('eval.ch.nist2005', 'r').readlines()
f = open('eval3.ch.nist2005', 'w')
for line in lines:
  line = line.replace(' ','')
  f.write(line)

lines = open('eval.en.nist2005', 'r').readlines()
ff = open('eval3.en.nist2005', 'w')
f = open('eval3.en.lower.nist2005', 'w')
for line in lines:
  line = line.replace('\n','')[1:-1] + '\n'
  ff.write(line)
  f.write(line.lower())