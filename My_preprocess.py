"""
Build the process of the pretraining and emtoion dataset
date: 2022/09/8
"""

import enum
import soundfile as sf
from transformers import Wav2Vec2Processor
import time
import re
import json
import joblib as job
from tqdm import tqdm
import unicodedata
import argparse
from io import open
import Const
from Utils import loadFrJoblib, saveToJoblib, saveToPickle, loadFrPickle, timeSince
from pdb import set_trace as stop
import numpy as np
import os
import torch
from torch.nn import functional as F
# from transformers import RobertaTokenizer, AutoTokenizer, HubertModel
from transformers import BertTokenizer, AutoTokenizer, HubertModel
# RobertaTokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# RobertaTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
RobertaTokenizer = BertTokenizer.from_pretrained(
    '../../pretrain_checkpoints/Bert')

Hubertprocessor = Wav2Vec2Processor.from_pretrained(
    "facebook/hubert-large-ls960-ft")
Hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


class my_process:
    def __init__(self, phaselist, opt) -> None:
        self.phaselist = phaselist
        self.opt = opt
        self.max_seq_len = opt.max_seq_len
        self.max_cont_len = opt.max_cont_len
        self.max_audio_max_len = opt.max_audio_max_len
        self.base_path = opt.base_path
        self.MELD_base = os.path.join(opt.base_path, opt.dataMELD)
        self.IEM_base = os.path.join(opt.base_path, opt.dataIEM)
        self.movie_base = os.path.join(opt.base_path, opt.dataPre)
        self.movie_audio = os.path.join(opt.base_path, opt.dataPreAud)
        # os.path.join(opt.base_path, opt.dataMELDAud)
        self.MELD_audio = opt.base_path
        self.IEM_audio = os.path.join(opt.base_path, opt.dataIEMAud)
        self.savePre = opt.savePre

    def movie_pre(self):
        moviefield = dict()
        for phase in self.phaselist:
            moviefield[phase] = self.movie_indexSrc(phase=phase)

        # saveToJoblib(self.savePre+'/moviefield_data.job', moviefield)
        torch.save(moviefield, self.savePre+'/moviefield_data.pt')
        # saveToPickle(self.savePre+'/moviefield_data.pkl', moviefield)
        print('moviefield data is saved!\n')

    def movie_indexSrc(self, phase):
        time_st = time.time()
        filename = self.movie_base + phase + '.job'
        data = job.load(filename)
        scripts = [scr['script'] for scr in data]
        negs = [scr['neg'] for scr in data]
        print('Processing file {}, length {}...'.format(filename, len(scripts)))

        scriptidxs = []
        scriptidxs_attmask = []
        audioinfo = []
        negidxs = []
        negidxs_attmask = []
        negaudio = []
        labelidxs = []
        neg_num = []
        count = 0
        wav_fea_paths_all = []
        newav_fea_paths_all = []
        for scr, ne in zip(scripts, negs):
            count += 1
            if count % 1000 == 0:
                print('-->{} dialogs {}'.format(timeSince(time_st), count))
            scridxs, scridxs_attmask, wav_feas, wav_fea_paths = self.movie_data_pre(
                scr)
            scriptidxs.append(scridxs)
            scriptidxs_attmask.append(scridxs_attmask)
            audioinfo.append(wav_feas)
            wav_fea_paths_all.append(wav_fea_paths)
            # negtive

            neidxs, neidxs_attmask, newav_feas, newav_fea_paths = self.movie_data_pre(
                ne)
            negidxs_attmask.append(neidxs_attmask)
            negidxs.append(neidxs)
            negaudio.append(newav_feas)
            newav_fea_paths_all.append(newav_fea_paths)
        neg_num = len(neidxs)
        scrfield = dict()

        scrfield['script'] = scriptidxs
        scrfield['script_attmask'] = scriptidxs_attmask
        scrfield['audio'] = audioinfo
        scrfield['neg'] = negidxs
        scrfield['neg_attmask'] = negidxs_attmask
        scrfield['negaudio'] = negaudio
        scrfield['neg_num'] = neg_num
        scrfield['label'] = labelidxs
        scrfield['audio_fea_path'] = wav_fea_paths_all
        scrfield['neaudio_fea_path'] = newav_fea_paths_all
        return scrfield

    def movie_data_pre(self, scr):
        scridxs = []
        wav_fea_paths = []
        scridxs_attmask = []
        wav_feas = []
        for item in scr:
            utterance = item["utterance"]
            audio_fea = item["audio"]  # here this may be None
            wav_path = item["wav_path"]
            w_wav_path = os.path.join(self.movie_audio, wav_path)

            # process utterance
            utt_idxs = RobertaTokenizer(utterance)

            input_ids = utt_idxs['input_ids']
            attention_mask = utt_idxs['attention_mask']

            if len(input_ids) < self.max_seq_len:
                input_ids = input_ids+[0]*(self.max_seq_len-len(input_ids))
                attention_mask = attention_mask + \
                    [0]*(self.max_seq_len-len(attention_mask))

            elif len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len-1]+[input_ids[-1]]
                attention_mask = attention_mask[:self.max_seq_len]

            # utt_scr.append(utterance)

            base_fea_path = os.path.join(
                self.savePre, 'movie_vectors',  wav_path.split('/')[-2])
            fea_path = os.path.join(base_fea_path,
                                    wav_path.split('/')[-1][:-4]+'.job')

            if not os.path.exists(base_fea_path):
                os.makedirs(base_fea_path)
            if not os.path.exists(fea_path):
                speech, _ = sf.read(w_wav_path)  # , sampling_rate=16000
                # , return_tensors="pt" .input_values
                wav_fea = Hubertprocessor(
                    speech, return_tensors="pt")
                saveToJoblib(fea_path, wav_fea)
            else:
                # put fea into folder
                wav_fea = loadFrJoblib(fea_path)

            # test_ = Hubert(wav_fea)
            wav_fea = None
            wav_fea_relpath = os.path.join(
                'movie_vectors',  wav_path.split('/')[-2], wav_path.split('/')[-1][:-4]+'.job')
            wav_fea_paths.append(wav_fea_relpath)
            # wav_fea = None
            # pad

            scridxs.append(input_ids)
            scridxs_attmask.append(attention_mask)
            wav_feas.append(wav_fea)
        return scridxs, scridxs_attmask, wav_feas, wav_fea_paths

    def MELD_pre(self):
        MELDfield = dict()
        meld_emo2idx, meld_senti2idx = self.get_label2idx(self.MELD_base)
        for phase in self.phaselist:
            filename = self.MELD_base + phase + '.job'
            MELDfield[phase] = self.MELD_indexSrc(
                phase=phase, filename=filename, MELD_audio=self.MELD_audio, emo2idx=meld_emo2idx, senti2idx=meld_senti2idx, decript='MELD_vectors')
        torch.save(MELDfield, self.savePre+'/MELDfield_data.pt')
        # saveToJoblib(self.savePre+'/MELDfield_data.job', MELDfield)
        # saveToJoblib(self.savePre+'/MELDfield_data.job', MELDfield)
        print('MELDfield data is saved!\n')

    def MELD_indexSrc(self, phase, filename, MELD_audio, emo2idx, senti2idx, decript):
        data = job.load(filename)
        diadata = data
        emodata = [[utter['next emotion']
                    for utter in dialog if 'next emotion' in utter.keys()] for dialog in data]
        sentidata = [[utter['next sentiment']
                      for utter in dialog if 'next sentiment' in utter.keys()] for dialog in data]
        print('Processing file {}, length {}...'.format(filename, len(diadata)))

        diaidxstxt = []
        diaidxs = []
        diaidxs_attmask = []
        diaauds = []
        emoidxs = []
        sentiidxs = []
        wav_fea_paths = []
        for dia in diadata:
            dia_idxs = []
            dia_idxs_attmask = []
            dia_aud = []
            dia_text = []
            relpaths = []
            for ditem in dia:
                utterance = ditem['utterance']
                speaker = ditem['speaker']
                wav_path = ditem['wav_path']
                emotion = ditem['emotion']
                sentiment = ditem['sentiment']
                audio = ditem['audio']

                # process
                utt_idxs = RobertaTokenizer(utterance)
                input_ids = utt_idxs['input_ids']
                attention_mask = utt_idxs['attention_mask']
                if len(input_ids) < self.max_seq_len:
                    input_ids = input_ids+[0]*(self.max_seq_len-len(input_ids))
                    attention_mask = attention_mask + \
                        [0]*(self.max_seq_len-len(attention_mask))

                elif len(input_ids) > self.max_seq_len:
                    input_ids = input_ids[:self.max_seq_len-1]+[input_ids[-1]]
                    attention_mask = attention_mask[:self.max_seq_len]

                w_wav_path = os.path.join(MELD_audio, wav_path)
                # speech, _ = sf.read(w_wav_path)  # , sampling_rate=16000
                # wav_fea = Hubertprocessor(
                #     speech, return_tensors="pt").input_values

                # decript
                base_fea_path = os.path.join(
                    self.savePre, decript, wav_path.split('/')[-2])
                fea_path = os.path.join(base_fea_path,
                                        wav_path.split('/')[-1][:-4]+'.job')

                if not os.path.exists(base_fea_path):
                    os.makedirs(base_fea_path)
                if not os.path.exists(fea_path):
                    speech, _ = sf.read(w_wav_path)  # , sampling_rate=16000
                    # , return_tensors="pt" .input_values
                    wav_fea = Hubertprocessor(
                        speech, return_tensors="pt")
                    saveToJoblib(fea_path, wav_fea)
                else:

                    # put fea into folder
                    try:
                        wav_fea = loadFrJoblib(fea_path)
                    except:
                        stop()
                wav_fea = None
                wav_fea_relpath = os.path.join(
                    decript, wav_path.split('/')[-2], wav_path.split('/')[-1][:-4]+'.job')
                relpaths.append(wav_fea_relpath)

                dia_idxs.append(input_ids)
                dia_idxs_attmask.append(attention_mask)
                dia_aud.append(wav_fea)
                dia_text.append(utterance)
            # cut context
            begin_cut = len(
                dia_idxs)-self.max_cont_len if len(dia_idxs) > self.max_cont_len else 0
            diaidxs.append(dia_idxs[begin_cut:].copy())
            diaidxs_attmask.append(dia_idxs_attmask[begin_cut:].copy())
            diaauds.append(dia_aud[begin_cut:].copy())
            diaidxstxt.append(dia_text[begin_cut:].copy())
            wav_fea_paths.append(relpaths[begin_cut:].copy())

        # emotion
        meld_emo2idx = emo2idx
        meld_senti2idx = senti2idx
        # emolist = list(set([utter['emotion']
        #                for dialog in data for utter in dialog]))
        # meld_emo2idx = {emo: i for i, emo in enumerate(emolist)}
        # sentilist = list(set([utter['sentiment']
        #                       for dialog in data for utter in dialog]))
        # meld_senti2idx = {senti: i for i, senti in enumerate(sentilist)}
        for emo in emodata:
            e_idxs = []
            for e in emo:
                e_idx = [meld_emo2idx[e]]
                e_idxs.append(e_idx)
            emoidxs.append(e_idxs)

        for senti in sentidata:
            s_idxs = []
            for s in senti:
                s_idx = [meld_senti2idx[s]]
                s_idxs.append(s_idx)
            sentiidxs.append(s_idxs)

        diafield = dict()

        diafield['feat'] = diaidxs
        diafield['feat_attmask'] = diaidxs_attmask
        diafield['audio'] = diaauds
        diafield['emo2idx'] = meld_emo2idx
        diafield['senti2idx'] = meld_senti2idx
        diafield['NextEmoLabel'] = emoidxs
        diafield['NextSentiLabel'] = sentiidxs
        diafield['text'] = diaidxstxt
        diafield['audio_fea_path'] = wav_fea_paths

        return diafield

    def get_label2idx(self, data_base):
        datas = []
        for phase in self.phaselist:
            filename = data_base + phase + '.job'
            data = job.load(filename)
            datas.append(data)
        emolist = list(set([utter['emotion']
                            for data in datas for dialog in data for utter in dialog]))
        meld_emo2idx = {emo: i for i, emo in enumerate(emolist)}
        sentilist = list(set([utter['sentiment']
                             for data in datas for dialog in data for utter in dialog]))
        meld_senti2idx = {senti: i for i, senti in enumerate(sentilist)}

        return meld_emo2idx, meld_senti2idx

    def IEM_pre(self):
        IEMfield = dict()
        meld_emo2idx, meld_senti2idx = self.get_label2idx(self.IEM_base)

        for phase in self.phaselist:
            filename = self.IEM_base + phase + '.job'
            IEMfield[phase] = self.MELD_indexSrc(
                phase=phase, filename=filename, MELD_audio=self.IEM_audio, emo2idx=meld_emo2idx, senti2idx=meld_senti2idx, decript='IEM_vectors')

        torch.save(IEMfield, self.savePre+'/IEMfield_data.pt')
        # saveToJoblib(self.savePre+'/IEMfield_data.job', IEMfield)
        # saveToPickle(self.savePre+'/IEMfield_data.pkl', IEMfield)
        print('IEMfield data is saved!\n')

    def IEM_indexSrc(self, phase):
        pass
        # data = job.load(filename)
        # stop()

    def run(self,):
        self.movie_pre()
        self.MELD_pre()
        self.IEM_pre()
        pass


if __name__ == "__main__":

    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-emoset', type=str)
    parser.add_argument('-min_count', type=int, default=2)
    parser.add_argument('-max_seq_len', type=int, default=60)
    parser.add_argument('-max_cont_len', type=int, default=11)
    parser.add_argument('-max_audio_max_len', type=int, default=200)  # 230
    parser.add_argument('-datatype', type=str, default='opsub')

    parser.add_argument('-base_path', type=str,
                        default='/data1/xcju/project_6/Whole_dataset')
    parser.add_argument('-dataMELD', type=str,
                        default='MELD_pairs_2/mfc74_sp_pairs_')
    parser.add_argument('-dataMELDAud', type=str,
                        default='')
    # parser.add_argument('-dataMELD_emo', type=str,
    #                     default='/data/xcju/code/dataset/MELD_pairs/mfc74_sp_pairs_')
    parser.add_argument(
        '-dataIEM', type=str, default='IEMOCAP_pairs_2/mfc74_sp_pairs_')
    parser.add_argument(
        '-dataIEMAud', type=str, default='IEMOCAP/IEMOCAP_full_release')
    parser.add_argument(
        '-dataPre', type=str, default='movie_predata/pre_data_acl/TvTextAudFeaPairs_mfc74_')
    parser.add_argument(
        '-dataPreAud', type=str, default='movie_predata')
    parser.add_argument(
        '-savePre', type=str, default='/data1/xcju/project_6/dataset/pre_MELD_IEM_bert')

    opt = parser.parse_args()

    print(opt, '\n')

    phaselist = ['train', 'dev', 'test']

    my_process(phaselist, opt).run()
    # process data

    # if opt.datatype == 'opsub':
    #     global_vocab(opt, phaselist=phaselist,
    #                  min_count=opt.min_count, max_seq_len=opt.max_seq_len)

    # if opt.datatype == 'emo':
    #     # dirt = 'Data/' + opt.emoset + '/' + opt.emoset + '_'
    #     dirt = opt.dataMELD
    #     # dirt = opt.dataIEM
    #     proc_emoset(opt, dirt=dirt, phaselist=phaselist, emoset=opt.emoset,
    #                 min_count=opt.min_count, max_seq_len=opt.max_seq_len)


# if __name__ == '__main__':

#     main()
