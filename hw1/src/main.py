#!/usr/env/bin python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from pandas import read_csv

from utils import *

MAX_ITERATION = 5
ALPHA = 0.01

def split(data):
  ''' split data and predicts '''
  return normalize(data[:, :-1]), data[:, -1]

def normalize(data):
  ''' normalize to [0, 1] '''
  return (data - data.min()) / data.max()

def hypothesis(case, args):
  return np.sum(case * args)

def save_args(args, dest):
  with open(dest, 'wb') as f:
    for item in args:
      f.write('{0},'.format(item))

def get_args(args_path, dimension=0):
  print args_path
  print os.path.exists(args_path)
  if os.path.exists(args_path):
    return np.array(read_csv(args_path, sep=',', header=None))[:, :-1]
  else:
    return np.full(dimension, 0, dtype=np.float)

def run_test(files):
  args = get_args(files.args)

  _test = normalize(np.array(read_csv(files.test, sep=',', header=0, index_col=0)))
  test = np.insert(_test, 0, 1., axis=1)

  with open(files.result, 'wb') as f:
    f.write('Id,reference\n')

    for idx, row in enumerate(test):
      f.write('{0},{1}\n'.format(idx, hypothesis(row, args)))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-train', action='store_true', help='Train arguments.')
  parser.add_argument('-test', action='store_true', help='Run test case with trained arguments.')
  parser.add_argument('-plot', action='store_true', help='Plot learning curve or not.')
  parser.add_argument('--method', type=int, choices=[1], required=True, help='[1 : gradient descent]')
  cli_args = parser.parse_args()

  files = get_filenames()

  if cli_args.train:
    data = np.array(read_csv(files.train, sep=',', header=0, index_col=0))
    _train, predict = split(data)

    train = np.insert(_train, 0, 1., axis=1)
    M, N = train.shape

    args = get_args(files.args, N)
    iteration = 0

    iters, J = [], []

    while iteration < MAX_ITERATION:
      iteration += 1
      print 'Iteration = %d' % iteration

      deviation = np.apply_along_axis(lambda row: hypothesis(row, args), 1, train) - predict
      cost = np.sum((deviation - predict) ** 2.) / (2. * M)

      J.append(cost)
      iters.append(iteration)

      _args = []
      for idx, theta in enumerate(args):
        _args.append(theta - np.sum(deviation * train[:, idx]) * ALPHA / (1. * M))

      args = np.array(_args)

    save_args(args, files.args)

    if cli_args.plot:
      plot(iters, J, 'Num of iteration', 'J', 'Gradient Descent', files.curve, alpha=ALPHA)

  if cli_args.test:
    run_test(files)

if __name__ == '__main__':
  main()