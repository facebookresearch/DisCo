# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import pickle, sys, torch

data = sys.argv[1]
with open(data, 'rb') as fin:
    data = pickle.load(fin)
rank = data[0]
outputs = data[1]
_, order = rank.sort()
max_lens = [0 for _ in range(len(outputs[0].split(' ')))]
for output in outputs:
    for i, max_len, token in zip(range(len(max_lens)), max_lens, output.split(' ')):
        token = str(int(rank[i])) + ':' + token
        if max_len < len(token):
            max_lens[i] = len(token)
print(max_lens)
        
for output in outputs:
    out = ''
    for i, max_len, token in zip(range(len(max_lens)), max_lens, output.split(' ')):
        out_token = '|{0: <' + str(max_len) + '}|'
        out_token = out_token.format(str(int(rank[i])) + ':' +  token)
        out += out_token
    print(out)
