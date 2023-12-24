import numpy as np
import torch 





def lengths_to_mask(lengths,seq_len):
    """
    @lengths: batch_size x 1
    return : batch_size x seq_len
    """
    # one_tensor = torch.ones(len(lengths),seq_len)
    # le_ts = torch.tensor(lengths).long().squeeze(1)
    # one_tensor[:,:lengths]=0
    return torch.LongTensor([[0 if i < le else 1 for i in range(seq_len)]for le in lengths.tolist()])
