from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

tf.enable_eager_execution()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

csv_path = 'msssc_20190930_20191227.csv'
df = pd.read_csv(csv_path)

model = tf.keras.models.load_model('simple_lstm.h5')
print(model.predict(20))