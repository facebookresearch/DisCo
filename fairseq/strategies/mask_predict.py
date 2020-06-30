# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


@register_strategy('mask_predict')
class MaskPredict(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
        self.move_eos = args.move_eos
    
    def generate(self, model, encoder_out, tgt_tokens, tgt_dict):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        eos_mask = tgt_tokens.eq(tgt_dict.eos()) 
        seq_lens = seq_len - pad_mask.sum(dim=1) - 1.0
        
        iterations = seq_len if self.iterations is None else self.iterations
        
        tgt_tokens, token_probs, decoder_out = self.generate_non_autoregressive(model, encoder_out, tgt_tokens)
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        assign_single_value_byte(tgt_tokens, eos_mask, tgt_dict.eos())
        assign_single_value_byte(token_probs, eos_mask, 1.0)

        for counter in range(1, iterations):
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            assign_single_value_byte(token_probs, eos_mask, 1.0)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            assign_single_value_byte(tgt_tokens, eos_mask, tgt_dict.eos())
            if self.move_eos:
                tgt_tokens_new = tgt_tokens.clone().fill_(tgt_dict.eos())
                tgt_tokens_new[:, 1:] = tgt_tokens[:, :-1]
                tgt_tokens = tgt_tokens_new
                tgt_tokens = tgt_tokens.masked_fill_(pad_mask, tgt_dict.pad())
            decoder_out = model.decoder(tgt_tokens, encoder_out, masking_type='token_masking')
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            
            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            assign_single_value_byte(token_probs, eos_mask, 1.0)
            
            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            assign_single_value_byte(tgt_tokens, eos_mask, tgt_dict.eos())
        lprobs = token_probs.log().sum(-1)

        return tgt_tokens, lprobs
    
    def generate_non_autoregressive(self, model, encoder_out, tgt_tokens):
        decoder_out = model.decoder(tgt_tokens, encoder_out, masking_type='token_masking')
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs, decoder_out

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)
