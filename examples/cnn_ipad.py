from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser('~'), 'scratchDL'))

import numpy as np
import gzip
import requests
import io

url = 'https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/data/digits.csv.gz?raw=true'

f = requests.get(url).content
data = np.loadtxt(gzip.open(io.BytesIO(f), 'rt'),
                  delimiter=',', dtype=np.float32)
