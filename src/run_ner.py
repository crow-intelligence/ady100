import subprocess
from os import listdir
from os.path import isfile, join

in_path = 'data/raw'
out_path = 'data/interim/ner'


def call_ner(f):
    print(f)
    subprocess.call("java -Xmx3G -jar etc/ner.jar  -mode predicate -input "
                    + join(in_path, f) + " -output " +
                    join(out_path, f[:-4])
                    + ".out",
                    stderr=subprocess.STDOUT,
                    shell=True)

call_ner('ady.txt')
