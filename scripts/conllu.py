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
text = data[1]
_, order = rank.sort()
output_conllu = ''
for i in range(len(text)):
    if rank[i] == 0:
        ## ROOT
        output_conllu += '\t'.join([str(i+1), text[i], '_', '_', '_', '_', '0', '0', '_', '_'])
    else:
        par_idx = int(order[rank[i]-1])
        output_conllu += '\t'.join([str(i+1), text[i], '_', '_', '_', '_', str(par_idx+1), str(int(rank[i])), '_', '_'])
    output_conllu += '\n'

print(output_conllu)
