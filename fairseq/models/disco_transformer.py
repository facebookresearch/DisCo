# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import (
    register_model, register_model_architecture
)
from fairseq.modules import MaskedMultiheadAttention
from fairseq.models.bert_seq2seq import (
    build_embedding, gelu, SelfTransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder, Transformer_nonautoregressive
)


@register_model('disco_transformer')
class DisCoTransformer(Transformer_nonautoregressive):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        #for ds in task.datasets.values():
        #    ds.target_is_source = True

        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024
        #if not task.args.dynamic_masking:
        #    # TODO: Completely move masking to the model for general purposes.
        #    raise RuntimeError('QMasking requires dynamic on-the-fly masking.')
        if not args.ignore_eos_loss:
            raise RuntimeError('Ignore eos loss to avoid the edge case of not attending to any.')

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, is_encoder=True, path=args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, is_encoder=True, path=args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, is_encoder=False, path=args.decoder_embed_path
            )


        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, args.encoder_embed_scale)
        decoder = SelfTransformerDecoderQMask(args, tgt_dict, decoder_embed_tokens, args.decoder_embed_scale)
        return DisCoTransformer(encoder, decoder)


    @staticmethod
    def add_args(parser):
        Transformer_nonautoregressive.add_args(parser)
        parser.add_argument('--at-only', action='store_true', default=False,
                            help='autoregressive only')
        parser.add_argument('--perm-only', action='store_true', default=False,
                            help='permutation only')
        parser.add_argument('--mix-masking', action='store_true', default=False,
                            help='Mix AT and full')
        parser.add_argument('--at-rm', action='store_true', default=False,
                            help='Mix AT and RM')
        parser.add_argument('--maskp', action='store_true', default=False,
                            help='Only one masking configuration for each sentence')



class SelfTransformerDecoderQMask(SelfTransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """
    def __init__(self, args, dictionary, embed_tokens, embed_scale=None, no_encoder_attn=False, left_pad=False,
                 final_norm=True):
        super().__init__(args, dictionary, embed_tokens, embed_scale, no_encoder_attn, left_pad, final_norm)
        # Update self.layers. This seems dirty. Refactorize the base init later.
        self.layers = nn.ModuleList([])
        if args.share_layers:
            self.layers.extend([
                TransformerDecoderQMaskLayer(args, no_encoder_attn)
            ])
            # number of times we run each decoder layer
            self.num_runs = args.decoder_layers
        else:
            # number of times we run each decoder layer
            self.num_runs = 1
            self.layers.extend([
                TransformerDecoderQMaskLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ])
        self.args = args

    def get_qmask(self, prev_output_tokens, decoder_padding_mask, masking_type, gen_order=None):
        # prev_output_tokens [B, T]
        # decoder_padding_mask [B, T]
        # return [B, T, T]
        # We softmax over the last dimension
        # Always attend to eos to avoid the all negative inf edge case.
        # Namely, [:, :, eos_idxes] = False (never mask out the attention to eos)
        bsz, max_len = decoder_padding_mask.shape
        if masking_type=='token_masking':
            q_mask = prev_output_tokens.eq(self.masking_idx).unsqueeze(1).repeat([1, max_len, 1])
            return q_mask

        elif masking_type=='full_masking':
            q_mask = prev_output_tokens.float().new_ones([bsz, max_len, max_len])
            q_mask = q_mask.bool()
            ## EOS is always available
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.eos_idx), False)
            return q_mask

        elif masking_type=='easy_first_masking':
            assert gen_order is not None
            # mask out yourself and later tokens
            ## EOS, BOS, and pad do not see any other token.
            q_mask = gen_order.unsqueeze(2) <= gen_order.unsqueeze(1)
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.eos_idx), True)
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.padding_idx), True)
            ## EOS is always available
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.eos_idx), False)
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.padding_idx), True)
            return q_mask

        assert masking_type=='random_masking'
        # never attend to yourself. You are predicting yourself. C.f. vanilla autoregressive
        q_mask = prev_output_tokens.float().new_zeros([bsz, max_len, max_len])
        # [bsz, max_len, max_len]
        # First generate uniform (0, 1) to determine which words to mask randomly
        if not self.training:
            # evaluation model. Use numpy seed to have the same masking configuration at each epoch.
            random_score = torch.Tensor(self.random.uniform(size = q_mask.shape)).to(q_mask.device)
            cutoff_ratio = torch.Tensor(self.random.uniform(size = [bsz, max_len])).to(q_mask.device)
        else:
            seed = 0
            self.random = np.random.RandomState(seed)
            random_score = q_mask.uniform_()
            cutoff_ratio = q_mask.new_zeros([bsz, max_len]).uniform_()
        ## eos and pad cannot see anyone so no information leakage.
        random_score = random_score.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.eos_idx), 5.0)
        random_score = random_score.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.padding_idx), 5.0)
        # Always mask pads. Put 5.0 so you always mask them.
        random_score = random_score.masked_fill_(decoder_padding_mask.unsqueeze(1), 5.0)
        # Always mask yourself. Put 5.0 so you always mask it.
        random_score = random_score.masked_fill_(torch.diag(q_mask.new_ones([max_len])).bool().unsqueeze(0), 5.0)
        ## We always unmask eos. So set them -5.0.
        random_score = random_score.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.eos_idx), -10.0)
        sorted_random_score, target_ordering = random_score.sort(2)
        _, target_rank = target_ordering.sort(2)
        target_lengths = max_len - decoder_padding_mask.float().sum(1, keepdim=True) - 1
        # -1 for eos.
        # target_lengths: [bsz, 1]
        cutoff_len = (target_lengths * cutoff_ratio).long() + 1
        # Cutoff_len chooses the number of unmasked words from [1, seq_length-1], including eos
        # Hardest case: we unmask only eos [1]
        # Easiest case: we unmask all but yourself [seq_length-1]
        q_mask = target_rank < cutoff_len.unsqueeze(2)
        # q_mask should be swapped. True for tokens that you DO NOT attend to.
        q_mask = ~q_mask
        return q_mask

    def forward(self, 
                prev_output_tokens,
                encoder_out=None,
                incremental_state=None,
                masking_type='random_masking',
                gen_order=None
                ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        incremental_state=None
        
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
        ) if self.embed_positions is not None else None

        x_kv = self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x_kv = self.project_in_dim(x)

        assert positions is not None
        x_kv += positions
        x_kv = F.dropout(x_kv, p=self.dropout, training=self.training)
        x_kv = x_kv.transpose(0, 1)
        x = positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        q_mask = self.get_qmask(prev_output_tokens, decoder_padding_mask, masking_type, gen_order)
        for layer in self.layers:
            for _ in range(self.num_runs):
                x, attn = layer(
                    x,
                    x_kv,
                    encoder_out['encoder_out'] if encoder_out is not None else None,
                    encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                    decoder_padding_mask,
                    q_mask,
                )
                if self.normalize:
                    x = self.layer_norm(x)
                inner_states.append(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None and self.load_softmax:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states, 'predicted_lengths': encoder_out['predicted_lengths']}


class TransformerDecoderQMaskLayer(TransformerDecoderLayer):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """
    def __init__(self, args, no_encoder_attn=False):
        super().__init__(args, no_encoder_attn)
        # Update self_attn with masked_multihead
        self.self_attn = MaskedMultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )

    def forward(self, x_q, x_kv, encoder_out, encoder_padding_mask, decoder_padding_mask, q_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x_q
        x_q = self.maybe_layer_norm(self.self_attn_layer_norm, x_q, before=True)
        x_kv = self.maybe_layer_norm(self.self_attn_layer_norm, x_kv, before=True)
        x_q, _ = self.self_attn(query=x_q, key=x_kv, value=x_kv, key_padding_mask=decoder_padding_mask, masked_attn=q_mask)
        # we do not use x_kv further. x_kv only knows output embedding input.
        x_q += residual
        x_q = F.dropout(x_q, p=self.dropout, training=self.training)
        x_q = self.maybe_layer_norm(self.self_attn_layer_norm, x_q, after=True)
        attn = None
        x = x_q
        # only run cross attention for x_q
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            #import IPython as ipy
            #ipy.embed()
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = gelu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn


@register_model_architecture('disco_transformer', 'disco_transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', args.encoder_embed_dim * 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', args.encoder_embed_dim // 64)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', args.encoder_attention_heads)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_enc_token_positional_embeddings = getattr(args, 'no_enc_token_positional_embeddings', False)
    args.no_dec_token_positional_embeddings = getattr(args, 'no_dec_token_positional_embeddings', False)
    args.embedding_only = getattr(args, 'embedding_only', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.decoder_embed_scale = getattr(args, 'decoder_embed_scale', None)
    args.encoder_embed_scale = getattr(args, 'encoder_embed_scale', None)

    args.bilm_mask_last_state = getattr(args, 'bilm_mask_last_state', False)
    args.bilm_add_bos = getattr(args, 'bilm_add_bos', False)


@register_model_architecture('disco_transformer', 'disco_transformer_big')
def bi_transformer_lm_big(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_architecture(args)
