#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import sys
import re
import os

K = 5

log = sys.argv[1]
checkpoint_dir = sys.argv[2]

epoch_re = re.compile(r'epoch ([0-9]+)')
def get_epoch(line):
    return int(epoch_re.search(line).group(1))

loss_re = re.compile(r'loss ([0-9\.]+)')
def get_loss(line):
    return float(loss_re.search(line).group(1))

with open(log) as fin:
    results = [(get_loss(line), get_epoch(line)) for line in fin if ' subset | loss' in line]

topk = [epoch for loss, epoch in sorted(results)[:K]]
print('TOP ' + str(K) + ':', topk)

cmd = 'python scripts/average_checkpoints.py --inputs ' + ' '.join([checkpoint_dir + '/checkpoint' + str(epoch) + '.pt' for epoch in topk]) + ' --output ' + checkpoint_dir + '/checkpoint_best_average.pt'
print(cmd)
os.system(cmd)

