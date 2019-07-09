import subprocess
from os import listdir
from os.path import isfile, join

in_path = 'data/raw'
out_path = 'data/interim/ml'


def call_magyarlanc(f):
    print(f)
    subprocess.call("java -Xmx1G -jar etc/magyarlanc-3.0.jar  -mode morphparse -input " + join(in_path, f) + " -output " + join(out_path, f[:-4]) + ".out", stderr=subprocess.STDOUT, shell=True)

call_magyarlanc('ady.txt')
