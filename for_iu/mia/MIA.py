import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

from torch.autograd import Variable
import numpy as np
from mia.Layers import RefineVisualLayer, RefineTextualLayer
from mia.SubLayers import MultiHeadedAttention_g

__author__ = "Fenglin Liu"
# Aligning Visual Regions and Textual Concepts for Semantic-Grounded Image Representations. NeurIPS 2019. #

class MIA(nn.Module):
    ''' Mutual Iterative Attention. '''
    
    def __init__(
           self,       
           n_head=8, d_k=64, d_v=64, d_model=512, d_inner=2048, N=2, dropout=0.1):

        super( MIA, self ).__init__()

        assert d_model == n_head * d_k and d_k == d_v
        self.N = N #  iteration times
        
        self.layer_refine_V = RefineVisualLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.layer_refine_T = RefineTextualLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.self_attn_g = MultiHeadedAttention_g(n_head, d_model, dropout)
        self.layer_norm = nn.LayerNorm( d_model )

    # def forward(self, V, T,G, return_attns=False):
    #
    #     # -- Forward
    #     Refine_V = V
    #     Refine_T = T
    #
    #     # Mutual Iterative Attention
    #     for i in range( self.N ):
    #         # Refining V
    #         Refine_V = self.layer_refine_V( Refine_T, Refine_V,G )
    #
    #         # Refining T
    #         Refine_T = self.layer_refine_T( Refine_V, Refine_T )
    #
    #     # SGIR = self.layer_norm( Refine_T + Refine_V ) # SGIR: Semantic-Grounded Image Representations
    #
    #     return Refine_V, Refine_T

    def forward(self, V, T, G,mask,return_attns=False):
        # -- Forward
        Refine_V = V
        Refine_T = T

        # Mutual Iterative Attention
        # for i in range(self.N):
            # Refining V
        G = self.self_attn_g(G, G, G, mask)
        Refine_V = self.layer_refine_V(Refine_T, Refine_V, G)

        # Refining T
        Refine_T = self.layer_refine_T(Refine_V, Refine_T)

        # SGIR = self.layer_norm( Refine_T + Refine_V ) # SGIR: Semantic-Grounded Image Representations

        Refine_V_1 = self.layer_refine_V(Refine_T, Refine_V, G)
        Refine_T_1 = self.layer_refine_T(Refine_V, Refine_T)

        Refine_V_2 = self.layer_refine_V(Refine_T_1, Refine_V_1, G)
        Refine_T_2 = self.layer_refine_T(Refine_V_1, Refine_T_1)

        return Refine_V_1, Refine_V_2
