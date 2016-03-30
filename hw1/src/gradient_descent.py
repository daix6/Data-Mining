#!/usr/env/bin python
# -*- coding: utf-8 -*-

import os, csv
import numpy as np

def normalize(data):
  ''' Normalize the data range
  '''
  datarange = np.amax(data, axis=0) - np.amin(data, axis=0)
  datamean = np.mean(data, axis=0)

  return np.nan_to_num((data - datamean) / datarange)

def hypothesis(case, args):
  ''' hypothesis to the regression of samples
  '''
  return np.sum(case * args[1:]) + args[0]

def gradient_descent(theta, deviation, feature, alpha=0.1):
  # print theta, deviation.shape, feature
  return theta - alpha * np.sum(deviation * feature) / deviation.shape[0]

def save_args(args, path):
  with open(path, 'wb') as f:
    w = csv.writer(f, dialect='excel')
    w.writerow(args)

def main():
  # Get data file position
  filepath = os.path.dirname(os.path.realpath(__file__))
  parent_dir = os.path.split(filepath)[0]
  data_dir = os.path.join(parent_dir, 'data')
  dest_dir = os.path.join(parent_dir, 'dest')

  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

  traindata_path = os.path.join(data_dir, 'train.csv')
  testdata_path = os.path.join(data_dir, 'test.csv')
  args_path = os.path.join(dest_dir, 'args.csv')
  result_path = os.path.join(dest_dir, 'output.csv')

  if not os.path.exists(args_path): # If there is not computed arguments, run gradient descent to compute
    with open(traindata_path, 'rb') as f:
      r = csv.reader(f, delimiter=',')
      d = np.asarray([i[1:] for i in r][1:], dtype=np.float) # remove header, and id in every case
      ref = np.asarray([i[-1] for i in d], dtype=np.float) # Get reference

    # print d[:, :-1]
    traindata = normalize(d[:, :-1]) # remove reference and normalize to (-1, 1)
    M, N = traindata.shape
    args = np.full((N + 1), 0, dtype=np.float) # N + 1(feature 0)
    args[0] = ref[0]

    # print traindata.shape, args.shape, ref.shape
    # print traindata, args, ref

    J = 1000

    while True:
      deviation = np.asarray([hypothesis(row, args) - ref[idx] for idx, row in enumerate(traindata)], dtype=np.float)
      cost = np.sum(deviation ** 2) / (2. * M)

      print cost

      if J - cost < 0.001: # Declare convergence if J(theta) decreases by less than 10^(-3) in one iteration.
        save_args(args, args_path)
        J = cost
        break

      J = cost if J > cost else J

      args_ = []
      args_.append(gradient_descent(args[0], deviation, np.full((M), 1, dtype=np.float))) # for x0
      for idx, theta in enumerate(args[1:]):
        args_.append(gradient_descent(theta, deviation, traindata[:, idx]))

      args = args_
  else: # Use computed args to run testcase
    with open(args_path, 'rb') as f:
      args = map(lambda x: np.float(x), f.read().split(','))

    # Run testcase
    with open(testdata_path, 'rb') as f:
      r = csv.reader(f, delimiter=',')
      d = np.asarray([i[1:] for i in r][1:], dtype=np.float) # Remove header, Id

    testdata = normalize(d)

    with open(result_path, 'wb') as f:
      f.write('Id,reference\n')

      for idx, row in enumerate(testdata):
        f.write('{0},{1}\n'.format(idx, hypothesis(row, args)))

if __name__ == '__main__':
  main()

