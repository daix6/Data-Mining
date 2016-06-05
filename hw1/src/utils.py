#!/usr/env/bin python
# -*- coding: utf-8 -*-

import os
from collections import namedtuple

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

DATA_DIR = 'data'
DEST_DIR = 'dest'

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
ARGS_FILE = 'result_args.csv'
RESULT_FILE = 'result.csv'
LEARNING_CURVE_FILE = 'learning-curve.png'

def get_filenames():
  file_dir = os.path.dirname(os.path.realpath(__file__))
  parent_dir, _ = os.path.split(file_dir)
  data_dir = os.path.join(parent_dir, DATA_DIR)
  dest_dir = os.path.join(parent_dir, DEST_DIR)

  filenames = map(lambda x : os.path.join(data_dir, x), [TRAIN_FILE, TEST_FILE]) \
            + map(lambda x : os.path.join(dest_dir, x), [ARGS_FILE, RESULT_FILE, LEARNING_CURVE_FILE])

  fn = namedtuple('Files', ['train', 'test', 'args', 'result', 'curve'])

  return fn(*filenames)

def plot(x, y, xlabel, ylabel, title, dest):
  plt.figure()
  plt.rc('font', family='serif')

  plt.grid(True)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  x, y = zip(*sorted(zip(x, y), key=lambda ref: ref[0]))

  # xnew = np.linspace(x[0], x[-1], 0.5)
  # ynew = interp1d(x, y)(xnew)

  # plt.plot(x, y, '.', xnew, ynew, '--')
  plt.plot(x, y, '-')

  plt.savefig(dest)
  print 'Save', title, 'to', dest
