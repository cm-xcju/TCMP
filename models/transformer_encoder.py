"""Base class for encoders and generic multi encoders."""
import torch
import torch.nn as nn
from models.sublayer import *
from models.myutils import *
# from utils.misc import aeq
# from models.sublayer import PositionwiseFeedForward
from pdb import set_trace as stop
class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerEncoderLayer, self).__init__()

    self.self_attn = MultiHeadedAttention(
        heads, d_model, dropout=dropout)#its self_attention
    
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)#PFF d_model->dff->dmodel use the relu
    
    self.att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs, mask):
    """
    @param inpust : seq_len x batch_size x dim
    @param mask :   batch_size x seq_len
    return : seq_len x batch_size x dim
    """
    input_norm = self.att_layer_norm(inputs)
    outputs, _ = self.self_attn(input_norm, input_norm, input_norm,
                                mask=mask)
    inputs = self.dropout(outputs) + inputs#add &norm
    
    input_norm = self.ffn_layer_norm(inputs)
    outputs = self.feed_forward(input_norm)
    inputs = outputs + inputs
    return inputs


class TransformerEncoder(nn.Module):

  def __init__(self, num_layers,input_dim, d_model, heads, d_ff,
               dropout ):# embeddings
    """

    :param num_layers: 6
    :param d_model: 512
    :param heads: 8
    :param d_ff: 2048
    :param dropout: 0.1
    :param embeddings: vocab_size->emb_size(512)
    """
    super(TransformerEncoder, self).__init__()

    self.num_layers = num_layers#6
    # self.embeddings = embeddings#vocab_size->emb_size
    self.transformer = nn.ModuleList(
      [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.linear_align = nn.Linear(input_dim, d_model)
    
    #position emb
    n_position=1000
    self.position_enc = nn.Embedding(n_position, d_model, padding_idx=0)
    self.position_enc.weight.data = position_encoding_init(n_position,d_model)

  def _check_args(self, src, lengths=None):
    _, n_batch,_ = src.size()
    if lengths is not None:
      n_batch_, = lengths.shape
      assert n_batch == n_batch_
    #   aeq(n_batch, n_batch_)

  def forward(self, src, lengths=None):
    """

    :param src: seq_len*bath_size* dim
    :param lengths:batch_size *1
    :return:seq_len*batch_size*dim
    """
    """ See :obj:`EncoderBase.forward()`"""
    
    src = self.linear_align(src)
    self._check_args(src, lengths)

    # emb = self.embeddings(src)
    # emb=src
    # out = emb.transpose(0, 1).contiguous()#bath_size*seq_len*emb_size[113*32*512]
    # words = src.transpose(0, 1)#bath_size*seq_len
    # padding_idx = self.embeddings.word_padding_idx#the position of the <bank>
    # mask = torch.zeros(src.size(1),1,src.size(0))#words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]  #in here I dont want to mask
    # Run the forward pass of every layer of the tranformer.
    length_matrix = lengths_to_mask(lengths,src.shape[0])
    mask = length_matrix.cuda().unsqueeze(1).bool()  # batch_size x 1 x t
    
    # position encoding
    pos_ids = torch.arange(src.shape[0]).unsqueeze(0).repeat(src.shape[1],1).cuda() # batch_size x seq_len
    pos_emb = self.position_enc(pos_ids).detach()# x batch_size x seq_len x dim

    out = pos_emb + src.transpose(0,1)

    for i in range(self.num_layers):
      out = self.transformer[i](out, mask)
    out = self.layer_norm(out)

    return  out

