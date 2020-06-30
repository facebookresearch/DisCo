# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# # This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


@register_strategy('easy_first')
class ParallelEasyFirst(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
        self.move_eos = args.move_eos
        self.counts = 0
        self.nb_sents = 0
        self.length_beam = args.length_beam 
    
    def generate(self, model, encoder_out, tgt_tokens, tgt_dict):
        bsz, seq_len = tgt_tokens.size()
        counting = [self.iterations for _ in range(int(bsz/self.length_beam))]
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        eos_mask = tgt_tokens.eq(tgt_dict.eos()) 
        seq_lens = seq_len - pad_mask.sum(dim=1) - 1.0
        
        iterations = seq_len if self.iterations is None else self.iterations
        
        tgt_tokens, token_probs, decoder_out = self.generate_non_autoregressive(model, encoder_out, tgt_tokens)
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        assign_single_value_byte(tgt_tokens, eos_mask, tgt_dict.eos())
        assign_single_value_byte(token_probs, eos_mask, 1.0)
        token_probs_orig = token_probs.clone()
        assign_single_value_byte(token_probs_orig, pad_mask, -200.0)
        assign_single_value_byte(token_probs_orig, eos_mask, -100.0)
        prev_converge_mask_sent = None

        for counter in range(1, iterations):
            if counter == 1:
                attn_mask = self.get_attn_mask(token_probs_orig, pad_mask, eos_mask)
                sorted_probs, sorted_ordering = token_probs.sort(1, descending=True)
                _, gen_order = sorted_ordering.sort(1)
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            assign_single_value_byte(tgt_tokens, eos_mask, tgt_dict.eos())
            decoder_out = model.decoder(tgt_tokens, encoder_out, masking_type='easy_first_masking', gen_order=gen_order)
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            
            # Handle eos and pads
            assign_single_value_byte(new_token_probs, pad_mask, 1.0)
            assign_single_value_byte(new_token_probs, eos_mask, 1.0)
            assign_single_value_byte(new_tgt_tokens, pad_mask, tgt_dict.pad())
            assign_single_value_byte(new_tgt_tokens, eos_mask, tgt_dict.eos())
            
            nb_sents = int(bsz/self.length_beam)
            new_tgt_tokens_reshaped = new_tgt_tokens.view(nb_sents, self.length_beam, seq_len)
            tgt_tokens_reshaped = tgt_tokens.view(nb_sents, self.length_beam, seq_len)
            new_token_probs = new_token_probs.view(nb_sents, self.length_beam, seq_len)
            non_eos_pad_mask = ~eos_mask.view(nb_sents, 
                self.length_beam, seq_len) & ~pad_mask.view(nb_sents, self.length_beam, seq_len)
            scores = new_token_probs.log().sum(2)/non_eos_pad_mask.sum(2).float()
            _, max_idxes = scores.max(1)
            # See if it converged
            converge_mask = new_tgt_tokens_reshaped.eq(tgt_tokens_reshaped).sum(2) == seq_len
            # See if the best sentence converged
            converge_mask_sent = converge_mask[torch.arange(nb_sents, device=converge_mask.device), max_idxes] > 0
            for batch_idx in range(nb_sents):
                if converge_mask_sent[batch_idx]:
                    counting[batch_idx] = min([counting[batch_idx], counter+1]) 

            if prev_converge_mask_sent is not None:
                # Only update scores for the sentences without convergence
                token_probs = token_probs.view(nb_sents, self.length_beam, seq_len)
                tgt_tokens = tgt_tokens.view(nb_sents, self.length_beam, seq_len)
                token_probs[~prev_converge_mask_sent] = new_token_probs[~prev_converge_mask_sent]
                tgt_tokens[~prev_converge_mask_sent] = new_tgt_tokens_reshaped[~prev_converge_mask_sent]
                token_probs = token_probs.view(bsz, seq_len)
                tgt_tokens = tgt_tokens.view(bsz, seq_len)
            else:
                token_probs = new_token_probs.view(bsz, seq_len)
                tgt_tokens = new_tgt_tokens
            if converge_mask_sent.all():
                # If every sentence converges, we are done.
                break
            prev_converge_mask_sent = converge_mask_sent

        lprobs = token_probs.log().sum(-1)
        self.nb_sents += len(counting)
        self.counts += sum(counting)

        return tgt_tokens, lprobs
    
    def generate_non_autoregressive(self, model, encoder_out, tgt_tokens):
        decoder_out = model.decoder(tgt_tokens, encoder_out, masking_type='token_masking')
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs, decoder_out

    def get_attn_mask(self, token_probs, pad_mask, eos_mask):
        sorted_probs, sorted_ordering = token_probs.sort(1, descending=True)
        _, gen_order = sorted_ordering.sort(1)
        q_mask = gen_order.unsqueeze(2) <= gen_order.unsqueeze(1)
        q_mask = q_mask.masked_fill_(eos_mask.unsqueeze(2), True)
        q_mask = q_mask.masked_fill_(pad_mask.unsqueeze(2), True)
        ## EOS is always available
        q_mask = q_mask.masked_fill_(eos_mask.unsqueeze(1), False)
        q_mask = q_mask.masked_fill_(pad_mask.unsqueeze(1), True)
        return q_mask
