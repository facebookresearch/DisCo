#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import re

def dehyphenate(sent):
    return re.sub(r'(\S)-(\S)', r'\1 ##AT##-##AT## \2', sent).replace('##AT##', '@')
with open('ende_at_1.0.pred', 'rt') as fin:
    sents = fin.readlines()
with open('ende_at_1.0_dehy.pred', 'wt') as fout:
    for sent in sents:
        fout.write(dehyphenate(sent))

