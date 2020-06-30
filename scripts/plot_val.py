# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import pickle

start = 91
end = start + 10
epochs = range(start, end)
epochs = [i*10 for i in epochs]
scores = pickle.load(open('data/test/test_dir/en-fr-1per/en-fr_epoch_{}-{}_10.pkl'.format(epochs[0], epochs[-1]), 'rb'))
print(epochs)
output = ['(' + str(int(epoch)) + ',' + str(bleu) + ')' for epoch, bleu in zip(epochs, scores)]
print(''.join(output))
