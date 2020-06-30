# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import (
    bleu,
    utils,
)
import numpy as np
import torch, pickle

def smoothed_sentence_bleu(dst_dict, target_tokens, hypo_tokens):
    """
    Implements "Smoothing 3" method from Chen and Cherry. "A Systematic
    Comparison of Smoothing Techniques for Sentence-Level BLEU".
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """
    scorer = bleu.Scorer(dst_dict.pad(), dst_dict.eos(), dst_dict.unk())
    scorer.add(target_tokens, hypo_tokens)

    invcnt = 1
    ratios = []
    for (match, count) in [
        (scorer.stat.match1, scorer.stat.count1),
        (scorer.stat.match2, scorer.stat.count2),
        (scorer.stat.match3, scorer.stat.count3),
        (scorer.stat.match4, scorer.stat.count4),
    ]:
        if count == 0:
            # disregard n-grams for values of n larger than hypothesis length
            continue
        if match == 0:
            invcnt *= 2
            match = 1.0 / invcnt
        ratios.append(match / count)

    brevity_penalty = np.min(
        [1, np.exp(1 - (scorer.stat.reflen / scorer.stat.predlen))]
    )
    geometric_mean = np.exp(np.log(ratios).mean())
    smoothed_bleu = brevity_penalty * geometric_mean
    return smoothed_bleu


if __name__ == '__main__':
    nb_greedy = 0
    nb_src = 0
    with open('tgt_dict.pkl', 'rb') as fin:
        tgt_dict = pickle.load(fin)
    for batch_id in range(188):
        with open('gold_{}.pkl'.format(batch_id), 'rb') as fin:
            gold = pickle.load(fin)
        with open('output_{}.pkl'.format(batch_id), 'rb') as fin:
            output = pickle.load(fin)
        with open('order_{}.pkl'.format(batch_id), 'rb') as fin:
            order = pickle.load(fin)
            order = order.reshape(output.size())
        # going over the beam
        for sent_id in range(output.size(0)):
            best_score = -10
            for j in range(output.size(1)):
                score = smoothed_sentence_bleu(tgt_dict, output[sent_id, j, :].int().cpu(), gold[sent_id].int().cpu())
                if  score > best_score:
                    best_score = score
                    best_tokens = output[sent_id, j, :]
                    best_idx = j
            best_tokens = utils.strip_pad(best_tokens, tgt_dict.pad())
            best_str = tgt_dict.string(best_tokens, '@@ ')
            best_order = order[sent_id, best_idx][:len(best_tokens)-1]
            greedy_order = order[sent_id, 0][:len(best_tokens)-1]
            if best_idx == 0:
                nb_greedy += 1
            else:
                print('Greedy: ', greedy_order)
                print('Best  : ', best_order)
            nb_src += 1
            #print(best_str)
            #best_tokens = utils.strip_pad(output[sent_id, 0, :], tgt_dict.pad())
            #best_tokens = utils.strip_pad(gold[sent_id], tgt_dict.pad())
            #best_str = tgt_dict.string(best_tokens, '@@ ')
            #print(best_str)
    print(nb_greedy)
    print(nb_src)
