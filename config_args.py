from email.policy import default
import os.path as path
import os
from pdb import set_trace as stop
# from tkinter.messagebox import NO
from typing import List
import argparse
import Const
import torch


def get_proprecess_args(parser):
    pass


def get_pretrain_args(parser):
    '''Main function'''

    # learning
    parser.add_argument('-lr', type=float, default=2e-4)
    parser.add_argument('-learning_rate_pretrained', type=float, default=2e-6)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-epochs', type=int, default=30)
    parser.add_argument('-patience', type=int, default=20,
                        help='patience for early stopping')
    parser.add_argument('-save_dir', type=str, default="snapshot",
                        help='where to save the models')
    # data
    parser.add_argument('-dataset', type=str, default='movie',
                        help='dataset')
    parser.add_argument('-savepath', type=str, default='/mnt/sda/xcju/project_6/dataset/pre_MELD_IEM',
                        help='dataset')
    parser.add_argument('-hubert', type=str, default='/mnt/sda/xcju/project_6/pretrain_checkpoints/Hubert/hubert-base-ls960',
                        help='dataset')
    parser.add_argument('-roberta', type=str, default='/mnt/sda/xcju/project_6/pretrain_checkpoints/roberta-base',
                        help='dataset')
    parser.add_argument('-bert', type=str, default='/mnt/sda/xcju/project_6/pretrain_checkpoints/Bert',
                        help='dataset')
    parser.add_argument('-use_bert', action='store_true',
                        help='how many steps to report loss')
    # parser.add_argument('-data_path', type=str, required=True,
    #                     help='data path')
    # parser.add_argument('-vocab_path', type=str, required=True,
    #                     help='global vocabulary path')
    parser.add_argument('-max_seq_len', type=int, default=60,
                        help='the sequence length')
    parser.add_argument('-num_neg', type=int, default=10,
                        help='the sequence length')
    parser.add_argument('-max_audio_max_len', type=int, default=200)  # 230
    parser.add_argument('-max_hubert_vec_maxsize',
                        type=int, default=80000)  # 230
    parser.add_argument('-accum_num', type=int, default=64,
                        help='how many steps to report loss')
    # model
    parser.add_argument('-sentEnc', type=str, default='Trans',
                        help='choose the low encoder')
    parser.add_argument('-contEnc', type=str, default='gru',
                        help='choose the mid encoder')
    parser.add_argument('-dec', type=str, default='revdec',
                        help='choose the classifier')
    parser.add_argument('-d_word_vec', type=int, default=768,  # 300
                        help='the word embeddings size')
    parser.add_argument('-d_audio_vec', type=int, default=768,  # 74,
                        help='the word embeddings size')
    parser.add_argument('-d_hidden_low', type=int, default=768,  # 300
                        help='the hidden size of rnn')
    parser.add_argument('-d_hidden_up', type=int, default=768,  # 300
                        help='the hidden size of rnn')
    parser.add_argument('-layers', type=int, default=1,
                        help='the num of stacked RNN layers')
    parser.add_argument('-fix_word_emb', action='store_true',
                        help='fix the word embeddings')
    parser.add_argument('-gpu', type=str, default=None,
                        help='gpu: default 0')
    parser.add_argument('-embedding', type=str, default=None,
                        help='filename of embedding pickle')
    parser.add_argument('-report_loss', type=int, default=720,
                        help='how many steps to report loss')
    parser.add_argument('-window', type=int, default=11,
                        help='how many steps to report loss')
    parser.add_argument('-winstride', type=int, default=5,
                        help='how many steps to report loss')
    parser.add_argument('-seed', type=int, default=42,
                        help='how many steps to report loss')
    parser.add_argument('-diff', type=str, default='0',
                        help='how many steps to report loss')
    parser.add_argument('-used_modalities', type=str, default='all',
                        help='how many steps to report loss')
    parser.add_argument('-Robert_hubert_pre', action='store_false',
                        help='how many steps to report loss')

    args = parser.parse_args()
    print(args, '\n')
    return args


def config_pretrain_args():
    parser = argparse.ArgumentParser()
    opt = get_pretrain_args(parser)
    if opt.use_bert:
        opt.savepath = opt.savepath+"_bert"
    opt.movie_checkpoint = os.path.join(opt.save_dir, str(opt.d_hidden_up) +
                                        "_"+str(opt.d_hidden_low)+"_"+opt.diff+"_"+opt.dataset+"_"+opt.used_modalities+".pt")
    opt.record_file = os.path.join(opt.save_dir, "movie_"+str(opt.d_hidden_up) +
                                   "_"+str(opt.d_hidden_low)+"_"+opt.diff+"_"+opt.dataset+"_"+opt.used_modalities+"_record.txt")

    opt.data_path = os.path.join(opt.savepath, 'moviefield_data.pt')
    opt.opt_save_path = os.path.join(
        opt.save_dir, 'pre_args_'+opt.used_modalities+'.pt')
    opt.train_score_file = os.path.join(
        opt.save_dir, 'train'+opt.used_modalities+'.txt')

    torch.save(opt, opt.opt_save_path)
    return opt


def get_train_args(parser):
    '''Main function'''
    # learning
    parser.add_argument('-lr', type=float, default=2e-6)
    parser.add_argument('-learning_rate_pretrained', type=float, default=2e-7)
    parser.add_argument('-decay', type=float, default=0.75)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-no_context', action='store_true')

    parser.add_argument('-epochs', type=int, default=60)
    parser.add_argument('-patience', type=int, default=50,
                        help='patience for early stopping')
    parser.add_argument('-pretrain_dir', type=str, default="snapshot",
                        help='where to save the models')
    parser.add_argument('-seed', type=int, default=42,
                        help='how many steps to report loss')
    parser.add_argument('-diff', type=str, default='0',
                        help='how many steps to report loss')
    # data
    parser.add_argument('-datasource', type=str,
                        default='/mnt/sda/xcju/project_6/dataset/pre_MELD_IEM', help='dataset')
    parser.add_argument('-hubert', type=str, default='/mnt/sda/xcju/project_6/pretrain_checkpoints/Hubert/hubert-base-ls960',
                        help='dataset')
    parser.add_argument('-roberta', type=str, default='/mnt/sda/xcju/project_6/pretrain_checkpoints/roberta-base',
                        help='dataset')
    parser.add_argument('-bert', type=str, default='/mnt/sda/xcju/project_6/pretrain_checkpoints/Bert',
                        help='dataset')
    parser.add_argument('-use_bert', action='store_true',
                        help='how many steps to report loss')

    parser.add_argument('-dataset', type=str, default='MELD',
                        help='dataset')
    # parser.add_argument('-data_path', type=str, required=True,
    #                     help='data path')

    # parser.add_argument('-emodict_path', type=str, required=True,
    #                     help='emotion label dict path')
    # parser.add_argument('-tr_emodict_path', type=str, default=None,
    #                     help='training set emodict path')
    parser.add_argument('-max_seq_len', type=int, default=60,  # 60 for emotion
                        help='the sequence length')

    # model
    parser.add_argument('-max_hubert_vec_maxsize',
                        type=int, default=80000)  # 230
    parser.add_argument('-sentEnc', type=str, default='gru2',
                        help='choose the low encoder')
    parser.add_argument('-contEnc', type=str, default='gru',
                        help='choose the mid encoder')
    parser.add_argument('-dec', type=str, default='dec',
                        help='choose the classifier')
    parser.add_argument('-d_word_vec', type=int, default=768,  # 300
                        help='the word embeddings size')
    parser.add_argument('-d_audio_vec', type=int, default=768,  # 74,
                        help='the word embeddings size')
    parser.add_argument('-d_hidden_low', type=int, default=768,  # 300
                        help='the hidden size of rnn')
    parser.add_argument('-d_hidden_up', type=int, default=768,  # 300
                        help='the hidden size of rnn')
    parser.add_argument('-layers', type=int, default=1,
                        help='the num of stacked GRU layers')
    parser.add_argument('-d_fc', type=int, default=100,
                        help='the size of fc')
    parser.add_argument('-gpu', type=str, default=None,
                        help='gpu: default 0')
    # parser.add_argument('-embedding', type=str, default=None,
    # help='filename of embedding pickle')
    parser.add_argument('-report_num', type=int, default=360,
                        help='how many steps to report loss')
    parser.add_argument('-accum_num', type=int, default=64,
                        help='how many steps to report loss')
    parser.add_argument('-save_model_path', type=str, default='',
                        help='how many steps to report loss')
    parser.add_argument('-load_model', action='store_true',
                        help='load the pretrained model')
    parser.add_argument('-used_modalities', type=str, default='all',
                        help='how many steps to report loss')
    parser.add_argument('-Robert_hubert_pre', action='store_false',
                        help='how many steps to report loss')

    args = parser.parse_args()
    print(args, '\n')
    return args


def config_train_args():

    parser = argparse.ArgumentParser()
    opt = get_train_args(parser)

    if opt.use_bert:
        opt.datasource = opt.datasource + "_bert"
    if opt.dataset in ['MELD_three', 'MELD']:
        opt.data_path = os.path.join(opt.datasource, 'MELDfield_data.pt')
    elif opt.dataset in ['IEMOCAP', 'IEMOCAP_four']:
        opt.data_path = os.path.join(opt.datasource, 'IEMfield_data.pt')
    if opt.dataset == 'MELD':
        opt.focus_emo = Const.sev_meld
    if opt.dataset == 'MELD_three':
        opt.focus_emo = Const.three_meld

    if opt.dataset == 'IEMOCAP':
        opt.focus_emo = Const.ten_iem
    if opt.dataset == 'IEMOCAP_four':
        opt.focus_emo = Const.four_iem
    opt.save_model_path = os.path.join('EmotionCheckpoint', opt.dataset)
    if not os.path.exists(opt.save_model_path):
        os.makedirs(opt.save_model_path)
    opt.load_pretrain_model = os.path.join(
        opt.pretrain_dir, str(opt.d_hidden_up) + "_"+str(opt.d_hidden_low)+"_0_movie"+"_"+opt.used_modalities+".pt")
    opt.emotioncheckpoint = os.path.join(opt.save_model_path, str(opt.d_hidden_up) +
                                         "_"+str(opt.d_hidden_low)+"_"+opt.diff+"_"+opt.dataset+"_"+opt.used_modalities+".pt")
    opt.savepath = opt.datasource
    opt.train_score_file = os.path.join(opt.save_model_path, str(opt.d_hidden_up) +
                                        "_"+str(opt.d_hidden_low)+"_"+opt.diff+"_"+opt.dataset+"_"+opt.used_modalities+"_record.txt")

    pre_args_path = os.path.join(
        opt.pretrain_dir, 'pre_args_'+opt.used_modalities+'.pt')
    # pre_args_path = os.path.join(
    #     opt.pretrain_dir, 'pre_args_'+opt.used_modalities+'.pt')
    pre_args = None
    if os.path.exists(pre_args_path):
        pre_args = torch.load(pre_args_path)

    if not opt.Robert_hubert_pre:
        opt.emotioncheckpoint = os.path.join(opt.save_model_path, str(opt.d_hidden_up) +
                                             "_"+str(opt.d_hidden_low)+"_"+opt.diff+"_"+opt.dataset+"_"+opt.used_modalities+"_pre_RoHuBert_"+str(opt.Robert_hubert_pre)+".pt")
        opt.train_score_file = os.path.join(opt.save_model_path, str(opt.d_hidden_up) +
                                            "_"+str(opt.d_hidden_low)+"_"+opt.diff+"_"+opt.dataset+"_"+opt.used_modalities+"_pre_RoHuBert_"+str(opt.Robert_hubert_pre)+"_record.txt")
    # if opt.load_model:

    pre_args.Robert_hubert_pre = opt.Robert_hubert_pre
    pre_args.hubert = opt.hubert
    pre_args.roberta = opt.roberta
    pre_args.bert = opt.bert
    pre_args.no_context = opt.no_context

    return opt, pre_args
#