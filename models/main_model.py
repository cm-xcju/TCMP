from socket import TCP_CORK
# from turtle import forward
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import Const
from models.sublayer import *
from models.transformer_encoder import *
from pdb import set_trace as stop
from transformers import BertModel, HubertModel,RobertaModel

class TCMP_pre(nn.Module):
    def __init__(self, args, num_layers=1) -> None:
        super(TCMP_pre, self).__init__()
        self.args = args

        # featrue encoding
        self.FeaEnc = FeatureEncoding(args=args)
        self.AT_CE = nn.CrossEntropyLoss()

        # for text audio multimodal
        self.for_text = modality_conlearning(args=args)
        self.for_audio = modality_conlearning(args=args)
        self.for_mm = modality_conlearning(args=args)

        self.gru_text = nn.GRU(input_size=args.d_word_vec, hidden_size=args.d_word_vec,
                               bidirectional=False, num_layers=num_layers, dropout=0.1)
        self.gru_audio = nn.GRU(input_size=args.d_word_vec, hidden_size=args.d_word_vec,
                                bidirectional=False, num_layers=num_layers, dropout=0.1)
        self.gru_Textaudio = nn.GRU(input_size=args.d_word_vec, hidden_size=args.d_word_vec,
                                    bidirectional=False, num_layers=num_layers, dropout=0.1)
        self.TA_align = nn.Linear(args.d_word_vec*2, args.d_word_vec)
        self.T_align = nn.Linear(args.d_word_vec*1, args.d_word_vec)
        self.A_align = nn.Linear(args.d_word_vec*1, args.d_word_vec)

    def forward(self, script, script_attmask, audio, audio_mask, neg_scr=None, neg_scr_mask=None, neg_audio=None, neg_audio_mas=None, state='pretrain'):

        if state == 'pretrain':
            T_enc, A_enc = self.FeaEnc(
                script, script_attmask, audio, audio_mask)
            T_enc_neg, A_enc_neg = self.FeaEnc(
                neg_scr, neg_scr_mask, neg_audio, neg_audio_mas)

            # loss  calculate wheter audio match text  [bach_size,1,d]
            AT_label = torch.LongTensor(
                [i for i in range(T_enc[:-1].shape[0])]).cuda()
            AT_score = torch.matmul(T_enc[:-1], A_enc[:-1].transpose(
                0, 1))  # * torch.exp(torch.tensor([-5]).float()).cuda()
            TA_score = torch.matmul(A_enc[:-1], T_enc[:-1].transpose(
                0, 1))  # * torch.exp(torch.tensor([-5]).float()).cuda()
            # Horizontal_prob = torch.softmax(AT_score.transpose(0, 1), 1)
            # Verical_prob = torch.softmax(AT_score, 1)
            h_loss = self.AT_CE(TA_score, AT_label)
            v_loss = self.AT_CE(AT_score, AT_label)

            # contextgru  target is not know
            T_enc = T_enc.unsqueeze(1)
            A_enc = A_enc.unsqueeze(1)
            T_enc_neg = T_enc_neg.unsqueeze(1)
            A_enc_neg = A_enc_neg.unsqueeze(1)
            T_all_gru_, _ = self.gru_text(T_enc)
            # T_all_gru = self.T_align(torch.cat([T_all_gru_, T_enc], -1))
            T_all_gru = self.T_align(T_all_gru_)
            A_all_gru_, _ = self.gru_audio(A_enc)
            # A_all_gru = self.A_align(torch.cat([A_all_gru_, A_enc], -1))
            A_all_gru = self.A_align(A_all_gru_)

            TA_enc = self.TA_align(torch.cat([T_enc, A_enc], -1))
            TA_enc_neg = self.TA_align(torch.cat([T_enc_neg, A_enc_neg], -1))
            TA_all_gru, _ = self.gru_Textaudio(TA_enc)
            T_all_gru = T_all_gru.squeeze(1)
            A_all_gru = A_all_gru.squeeze(1)
            TA_all_gru = TA_all_gru.squeeze(1)
            # for text
            loss_t, pred_t = self.for_text(T_all_gru, T_enc, T_enc_neg)
            loss_a, pred_a = self.for_audio(A_all_gru, A_enc, A_enc_neg)
            loss_mm, pred_mm = self.for_mm(TA_all_gru, TA_enc, TA_enc_neg)

            # # for Audio

            # utt_text_ = self.gru_text(T_enc)
            # utt_audio_ = self.gru_audio(A_enc)
            if self.args.used_modalities == 'all':
                loss = (h_loss+v_loss)/2 + (loss_t+loss_a + loss_mm)/3
                return loss, pred_t, pred_a, pred_mm
            elif self.args.used_modalities == 'audio':
                loss = (h_loss+v_loss)/2 + loss_a
                return loss, pred_t, pred_a, pred_mm
            elif self.args.used_modalities == 'text':
                loss = (h_loss+v_loss)/2 + loss_t
                return loss, pred_t, pred_a, pred_mm
            elif self.args.used_modalities == 'text2audio':
                loss = (h_loss+v_loss)/2 + loss_mm
                return loss, pred_t, pred_a, pred_mm
        elif state == 'emotion':
            T_enc, A_enc = self.FeaEnc(
                script, script_attmask, audio, audio_mask)
            T_enc = T_enc.unsqueeze(1)
            A_enc = A_enc.unsqueeze(1)
            T_all_gru_, _ = self.gru_text(T_enc)
            T_all_gru = self.T_align(T_all_gru_)
            # T_all_gru = self.T_align(torch.cat([T_all_gru_, T_enc], -1))
            A_all_gru_, _ = self.gru_audio(A_enc)
            A_all_gru = self.A_align(A_all_gru_)
            # A_all_gru = self.A_align(torch.cat([A_all_gru_, A_enc], -1))
            TA_enc = self.TA_align(torch.cat([T_enc, A_enc], -1))
            TA_all_gru, _ = self.gru_Textaudio(TA_enc)
            T_all_gru = T_all_gru.squeeze(1)
            A_all_gru = A_all_gru.squeeze(1)
            TA_all_gru = TA_all_gru.squeeze(1)
            repr_T = self.for_text(T_all_gru, T_enc, state=state)
            repr_A = self.for_audio(A_all_gru, A_enc, state=state)
            repr_MM = self.for_mm(TA_all_gru, TA_enc, state=state)
            return repr_T, repr_A, repr_MM


class modality_conlearning(nn.Module):
    def __init__(self, args):
        super().__init__()
        # context
        self.args = args
        self.cont_query = nn.Sequential(
            nn.Linear(args.d_word_vec*2, args.d_word_vec),
            nn.Tanh()
        )
        self.leftgru = nn.GRU(input_size=args.d_word_vec, hidden_size=args.d_word_vec,
                              bidirectional=False, num_layers=1, dropout=0.2)
        self.get_cont = nn.Linear(args.d_word_vec * 2, args.d_word_vec)
        self.Truefalse_CE = nn.CrossEntropyLoss()
        # self.Truefalse_CE = nn.BCEWithLogitsLoss()

    def forward(self, T_all_gru, T_enc, T_enc_neg=None, state='pretrain'):
        if state == 'pretrain':

            T_dia_len = T_all_gru.shape[0]
            t_context_right = T_all_gru[1:T_dia_len-1]
            t_context_source = T_all_gru[0:T_dia_len-2]
            querys = self.cont_query(
                torch.cat([t_context_right, t_context_source], 1))  # N-2 x dim
            one_tensor = torch.ones(T_dia_len-2, T_dia_len).triu()  # N-2 x N
            mask = one_tensor.bool().cuda()

            t_context = T_all_gru.unsqueeze(1)  # 1*N*dim
            left_cont, _ = self.leftgru(t_context)  # 1* N *dim
            left_cont = left_cont.squeeze(1)  # N *dim

            scores = torch.matmul(querys, left_cont.transpose(0, 1))  # N-2 x N
            scores = scores.masked_fill(mask, -1e18)
            att = torch.softmax(scores, 1) * (mask*(-1)+1)
            left_embed = torch.matmul(att, left_cont)  # N-2 xdim
            cont_embed = self.get_cont(
                torch.cat([left_embed, querys], 1)).unsqueeze(1)  # N-2  x 1 x d
            t_embed_ = T_enc[2:T_dia_len]  # N-2 x 1 x d
            # true_label = torch.matmul(cont_embed, t_embed_.transpose(1,2)) # N-2 x N-2
            # T_enc_neg= T_enc_neg.unsqueeze(0)

            true_label = torch.matmul(cont_embed, t_embed_.transpose(1, 2))
            f_embed = T_enc_neg.transpose(0, 1)  # Fx1xd
            f_embed_ = f_embed.expand(
                [t_embed_.size()[0], f_embed.size()[1], f_embed.size()[2]])
            neg_label = torch.matmul(cont_embed, f_embed_.transpose(1, 2))
            all_label_prob = torch.cat(
                [true_label, neg_label], dim=-1).squeeze(1)
            # la_idxs = [1] + [0] * T_enc_neg.shape[1]
            # laidxs = [la_idxs] * (T_dia_len - 2)
            ground_label = torch.LongTensor(
                [0 for i in range(all_label_prob.shape[0])]).cuda()
            # ground_label=F.one_hot(ground_label,all_label_prob.shape[1])

            # loss_pre = self.Truefalse_CE(all_label_prob, ground_label.float())
            loss_pre = self.Truefalse_CE(all_label_prob, ground_label)

            # f_embed_ = T_enc_neg.expand(
            #     [t_embed_.shape[1], T_enc_neg.shape[1], T_enc_neg.shape[2]])  # N-2 x F x d
            # true_false_feas = torch.cat(
            #     [t_embed_.transpose(1, 0), f_embed_], 1)  # N-2 x (F+1) x d
            # all_label = torch.matmul(cont_embed.unsqueeze(
            #     1), true_false_feas.transpose(1, 2)) * torch.exp(torch.tensor([-5]).float()).cuda()  # N-2 x (F+1)
            # all_label_prob = all_label.squeeze(1)   # N-2 x (F+1)
            # all_pred = all_label_prob.max(1)[1]
            # main_label = torch.LongTensor(
            #     [0 for i in range(true_false_feas.shape[0])]).cuda()
            # loss_pre = self.Truefalse_CE(all_label_prob, main_label)

            return loss_pre, all_label_prob
        elif state == 'emotion':

            T_dia_len = T_all_gru.shape[0]
            t_context_right = T_all_gru[-2].unsqueeze(0)
            t_context_source = T_all_gru[-3].unsqueeze(0)
            query = self.cont_query(
                torch.cat([t_context_right, t_context_source], 1))  # 1 x dim
            # t_context = T_all_gru[:T_dia_len-3].unsqueeze(0)  # 1*N-3*dim
            if self.args.no_context:
                return query
            t_context = T_all_gru.unsqueeze(1)  # 1*N-3*dim
            left_cont, _ = self.leftgru(t_context)  # 1* N-3 *dim
            left_cont = left_cont.squeeze(1)  # N-3 *dim
            scores = torch.matmul(query, left_cont.transpose(0, 1))  # 1 x N-3
            one_tensor = torch.ones(scores.shape[0], T_dia_len)  # N-2 x N
            one_tensor[:, :-3] *= 0

            mask = one_tensor.bool().cuda()
            scores = scores.masked_fill(mask, -1e18)
            att = torch.softmax(scores, 1) * (mask*(-1)+1)  # 1 x N-3

            left_embed = torch.matmul(att, left_cont)  # 1 xdim

            cont_embed = self.get_cont(
                torch.cat([left_embed, query], 1))  # 1  x d

            return cont_embed


class FeatureEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # featrue encoding
        
        if args.Robert_hubert_pre:
            # self.roberta = RobertaModel.from_pretrained(args.roberta)
            self.roberta = BertModel.from_pretrained(args.bert)
            self.hubert = HubertModel.from_pretrained(args.hubert)
        else:
            # self.roberta = RobertaModel()
            self.roberta = BertModel()
            self.hubert = HubertModel()
        self.dropout_0 = nn.Dropout(0.1)
        # self.Tnorm = nn.LayerNorm(args.d_word_vec)
        # self.Anorm = nn.LayerNorm(args.d_word_vec)
        self.output1 = nn.Sequential(
            nn.Linear(args.d_word_vec, args.d_word_vec),
            nn.Tanh()
        )

    def forward(self, script, script_attmask, audio, audio_mask):
        # encoding
        # [batch)size * seq_len * d]
        # for bert token_type_ids=script*0,
        bert_enc = self.roberta(
            input_ids=script, attention_mask=script_attmask).last_hidden_state
        hubert_enc = self.hubert(
            input_values=audio.mean(-1), attention_mask=audio_mask).last_hidden_state

        utt_text_state = (torch.max(bert_enc, dim=1)[
            0] + torch.mean(bert_enc, dim=1))/2
        utt_audio_state = (torch.max(hubert_enc, dim=1)[
            0] + torch.mean(hubert_enc, dim=1))/2
        utt_text_state = self.dropout_0(self.output1(utt_text_state))
        utt_audio_state = self.dropout_0(self.output1(utt_audio_state))

        # return self.Tnorm(utt_text_state), self.Anorm(utt_audio_state)
        return utt_text_state, utt_audio_state


class TCMP(nn.Module):
    def __init__(self, args, pre_args, pre_model=None, criterion=None) -> None:
        super(TCMP, self).__init__()

        self.pre_model = TCMP_pre(args=pre_args)
        if pre_model:
            self.pre_model.load_state_dict(
                pre_model.state_dict())

        self.args = args
        self.out_len = len(args.focus_emo)

        self.emotion_MLP = nn.Sequential(
            nn.Linear(args.d_word_vec*3, args.d_word_vec*2),
            nn.ReLU(),
            nn.Linear(args.d_word_vec*2, args.d_word_vec*1),
            nn.Linear(args.d_word_vec, self.out_len)

        )
        self.CSE = criterion
        # nn.BCEWithLogitsLoss()
        # self.CSE = nn.CrossEntropyLoss()

    def forward(self, feat, feat_attmask, audio, audio_mask, label):

        repr_T, repr_A, repr_MM = self.pre_model(
            feat, feat_attmask, audio, audio_mask, state='emotion')
        if self.args.used_modalities == 'text':
            emo_prob = self.emotion_MLP(torch.cat([repr_T, repr_A*0, repr_MM*0], 1))
        elif self.args.used_modalities == 'audio':
            emo_prob = self.emotion_MLP(torch.cat([repr_T*0, repr_A, repr_MM*0], 1))
        elif self.args.used_modalities == 'text2audio':
            emo_prob = self.emotion_MLP(torch.cat([repr_T*0, repr_A*0, repr_MM], 1))
        elif self.args.used_modalities == 'all':
            emo_prob = self.emotion_MLP(torch.cat([repr_T, repr_A, repr_MM], 1))
        else:
            raise ValueError(" not found used_modalities")

        # label_onehot = F.one_hot(label, self.out_len)
        loss = self.CSE(emo_prob, label.squeeze(0))
        # loss = self.CSE(emo_prob, label_onehot.squeeze(0).float())

        return loss, emo_prob
