"""
Train on Emotion dataset.
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Utils
import math
from pdb import set_trace as stop
from tqdm import tqdm, trange
from datetime import datetime

# def emotrain(wordenc, sentenc, contenc, cmdec, dec, audioenc, contAudenc, cmdecAud, data_loader, tr_emodict, emodict, args, focus_emo):


def emotrain(model, criterion, optimizer, data_loader, args):
    """
    :data_loader input the whole field
    """

    # start time
    time_st = time.time()
    decay_rate = args.decay
    # dataloaders
    train_loader = data_loader['train']
    dev_loader = data_loader['dev']
    test_loader = data_loader['test']

    feats, feat_attmasks, audio_paths, emo2idx, senti2idx = train_loader['feat'], train_loader[
        'feat_attmask'], train_loader['audio_fea_path'], train_loader['emo2idx'], train_loader['senti2idx']
    if args.dataset in ['MELD', 'IEMOCAP']:
        labels = train_loader['NextEmoLabel']
        label2idx = emo2idx
    elif args.dataset in ['MELD_three', 'IEMOCAP_four']:
        labels = train_loader['NextSentiLabel']
        label2idx = senti2idx

    lr = args.lr
    criterion.cuda()
    model.cuda()
    model.train()

    # lr_w, lr_s, lr_c, lr_d = [args.lr] * 4
    # # if args.load_model:
    # #	lr_c /= 2
    # #	lr_s /= 2
    # #	lr_w /= 2
    # wordenc_opt = optim.Adam(wordenc.parameters(), lr=lr_w)
    # sentenc_opt = optim.Adam(sentenc.parameters(), lr=lr_s)
    # contenc_opt = optim.Adam(contenc.parameters(), lr=lr_c)
    # cmdec_opt = optim.Adam(cmdec.parameters(), lr=lr_c)
    # dec_opt = optim.Adam(dec.parameters(), lr=lr_d)

    # # audio
    # audioenc_opt = optim.Adam(audioenc.parameters(), lr=lr_s)
    # contAudenc_opt = optim.Adam(contAudenc.parameters(), lr=lr_c)
    # cmdecAud_opt = optim.Adam(cmdecAud.parameters(), lr=lr_c)

    # weight for loss
    weight_rate = 0.5
    # if args.dataset in ['MOSI', 'IEMOCAP4v2']:
    #	weight_rate = 0
    # print(label2idx)
    # weights = torch.from_numpy(loss_weight(
    #     tr_emodict, emodict, focus_emo, rate=weight_rate)).float()
    # weights = torch.tensor([1 for i in range(emodict.n_words)]).float()
    # print("Dataset {} Weight rate {} \nEmotion rates {} \nLoss weights {}\n".format(
    #     args.dataset, weight_rate, emodict.word2count, weights))
    # weights[0]=0
    # criterion = torch.nn.CrossEntropyLoss(weights).cuda()
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    # wordenc.train()
    # sentenc.train()
    # contenc.train()
    # cmdec.train()
    # dec.train()

    # audioenc.train()
    # contAudenc.train()
    # cmdecAud.train()

    # wordenc.cuda()
    # sentenc.cuda()
    # contenc.cuda()
    # cmdec.cuda()
    # dec.cuda()

    # audioenc.cuda()
    # contAudenc.cuda()
    # cmdecAud.cuda()

    over_fitting = 0
    cur_best = -1e10
    glob_steps = 0
    report_loss = 0

    for epoch in range(1, args.epochs + 1):

        feats, feat_attmasks, audio_paths, labels = Utils.shuffle_lists(
            feats, feat_attmasks, audio_paths, labels)
        print("===========Epoch:{}==============".format(epoch))
        print("-{}-{}".format(epoch, Utils.timeSince(time_st)))
        for bz in range(len(feats)):
            # for bz in tqdm(range(len(feats)), mininterval=0.5, desc='(Training)', leave=False):
            # tensorize a dialog list
            # if bz > 5:
            #     continue
            if feats[bz] == []:
                continue
            feat, lens = Utils.ToTensor(feats[bz], is_len=True)
            feat_attmask, lens_attmasks = Utils.ToTensor(
                feat_attmasks[bz], is_len=True)
            # audio, aud_lens = Utils.ToAudioTensor(audios[bz], is_len=True)

            audio, audio_mask = Utils.AudiopathToTensor(
                audio_paths[bz], args=args, is_len=True)
            # negative sampling
            label = Utils.ToTensor(labels[bz])
            # feat = Variable(feat)
            # label = Variable(label)

            # if args.gpu != None:
            # 	# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            # 	# device = torch.device("cuda: 0")
            # 	wordenc.cuda()
            # 	sentenc.cuda()
            # 	contenc.cuda()
            # 	dec.cuda() xcju@970124

            # 	audioenc.cuda()
            # 	contAudenc.cuda()
            # due to that emotion utterance is incoprated, the final utterance has been abondoned in the pre_model.

            # if feat.shape[0] == 3:

            #     feat = torch.cat(
            #         [torch.zeros(1, feat.shape[1]).long(), feat], 0)
            #     feat_attmask = torch.cat(
            #         [torch.zeros(1, feat_attmask.shape[1]).long(), feat_attmask], 0)
            #     audio = torch.cat(
            #         [torch.zeros(1, audio.shape[1], audio.shape[2]), audio], 0)
            #     audio_mask = torch.cat(
            #         [torch.zeros(1, audio_mask.shape[1]).int(), audio_mask], 0)
            feat = feat.cuda()
            feat_attmask = feat_attmask.cuda()
            audio = audio.cuda()
            audio_mask = audio_mask.cuda()
            label = label.cuda()
            # weights = weights.cuda()

            loss, pred_prob = model(
                feat, feat_attmask, audio, audio_mask, label)

            # w_embed = wordenc(feat)
            # s_embed = sentenc(w_embed, lens)[0]

            # audio_scr = audioenc(audio, aud_lens)[0]
            # # contAud_scr = contAudenc(audio_scr)

            # # for no context
            # sr_text = s_embed[-2] + s_embed[-3]
            # sr_audio = audio_scr[-2] + audio_scr[-3]

            # s_dec_context = sr_text.unsqueeze(0)
            # contAud_scr_dec = sr_audio.unsqueeze(0)

            # #log_prob = dec(s_embed)
            # # s_context = contenc(s_embed)
            # s_context = s_embed

            # s_dec_context = cmdec(t_context=s_context)
            # # s_concat = torch.cat([s_embed, s_context], dim=-1)

            # contAud_scr = audio_scr
            # contAud_scr_dec = cmdecAud(t_context=contAud_scr)
            # # a_concat = torch.cat([audio_scr,contAud_scr],dim=-1)
            # # a_concat = torch.cat([s_dec_context,contAud_scr_dec],dim=-1)

            # log_prob = dec(s_dec_context, contAud_scr_dec)
            # log_prob = log_prob[-1].unsqueeze(0)
            # log_prob = dec(s_concat,a_concat)
            #log_prob = dec(s_context)
            #print(log_prob, label)
            # loss_n = comput_class_loss(log_prob, label, weights)

            # loss = criterion(log_prob, label.squeeze(1))
            # print('loss:{} ,pred:{}',loss.item(),torch.softmax(log_prob,1))

            loss.backward()
            report_loss += loss.item()
            glob_steps += 1

            # gradient clip
            # torch.nn.utils.clip_grad_norm_(wordenc.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(sentenc.parameters(), max_norm=5)
            # # torch.nn.utils.clip_grad_norm_(contenc.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(cmdec.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=5)

            # torch.nn.utils.clip_grad_norm_(audioenc.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(contAudenc.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(cmdecAud.parameters(), max_norm=5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            # wordenc_opt.step()
            # sentenc_opt.step()
            # # contenc_opt.step()
            # cmdec_opt.step()
            # dec_opt.step()

            # audioenc_opt.step()
            # # contAudenc_opt.step()
            # cmdecAud_opt.step()
            # The gradient accumulate
            if glob_steps % args.accum_num == 0:

                optimizer.step()
                optimizer.zero_grad()

            # wordenc_opt.zero_grad()
            # sentenc_opt.zero_grad()
            # # contenc_opt.zero_grad()
            # dec_opt.zero_grad()
            # cmdec_opt.zero_grad()

            # audioenc_opt.zero_grad()
            # # contAudenc_opt.zero_grad()
            # cmdecAud_opt.zero_grad()

            if glob_steps % args.report_num == 0:
                print("Steps: {} Loss: {} LR: {}".format(
                    glob_steps, report_loss/args.report_num, optimizer.param_groups[0]['lr']))
                report_loss = 0
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            # validate
            pAccs = emoeval(model=model, data_loader=dev_loader, args=args)
            print("dev: ACCs-F1s-WA-UWA-F1-Prec-Rec-miF1val {}".format(pAccs))
            pAccs_test = emoeval(
                model=model, data_loader=test_loader, args=args)
            print("Test: ACCs-F1s-WA-UWA-F1-Prec-Rec-miF1val {}".format(pAccs_test))
            # pAccs_train = emoeval(
            #     model=model, data_loader=train_loader, args=args)
            # print("train: ACCs-F1s-WA-UWA-F1-Prec-Rec-miF1val {}".format(pAccs_train))
        with open(args.train_score_file, 'a+') as ft:
            ft.write("epoch is {} \n".format(epoch))
            ft.write("Dev: ACCs-F1s-WA-UWA-F1-Prec-Rec-miF1val {} \n".format(pAccs))
            ft.write(
                "Test: ACCs-F1s-WA-UWA-F1-Prec-Rec-miF1val {} \n".format(pAccs_test))
            # ft.write(
            # "Train: ACCs-F1s-WA-UWA-F1-Prec-Rec-miF1val {} \n\n".format(pAccs_train))

        last_best = pAccs[2]
        # if args.dataset in ['MOSI', 'IEMOCAP4v2']:
        #	last_best = pAccs[-2]
        print("best={}".format(last_best))
        if last_best > cur_best:
            Utils.scrmodel_saver_new(model, args.emotioncheckpoint)

            # Utils.revmodel_saver(wordenc, args.save_model_path,
            #                      'wordenc', args.dataset, args.load_model)
            # Utils.revmodel_saver(sentenc, args.save_model_path,
            #                      'sentenc', args.dataset, args.load_model)
            # Utils.revmodel_saver(contenc, args.save_model_path,
            #                      'contenc', args.dataset, args.load_model)
            # Utils.revmodel_saver(dec, args.save_model_path,
            #                      'dec', args.dataset, args.load_model)
            # Utils.revmodel_saver(cmdec, args.save_model_path,
            #                      'cmdec', args.dataset, args.load_model)

            # Utils.revmodel_saver(audioenc, args.save_model_path,
            #                      'audioenc', args.dataset, args.load_model)
            # Utils.revmodel_saver(
            #     contAudenc, args.save_model_path, 'contAudenc', args.dataset, args.load_model)
            # Utils.revmodel_saver(cmdecAud, args.save_model_path,
            #                      'cmdecAud', args.dataset, args.load_model)

            # Utils.revmodel_saver(
            #     criterion, args.save_model_path, 'criterion', args.dataset, args.load_model)

            cur_best = last_best
            over_fitting = 0
        else:
            over_fitting += 1

            decay_rate = args.decay  # 0.5
            if optimizer.param_groups[0]['lr'] < 2e-6:
                decay_rate = 1.0
            if epoch > 5 and over_fitting > 2:
                optimizer.param_groups[0]['lr'] *= decay_rate
                # optimizer.param_groups[1]['lr'] *= args.decay
                # optimizer.param_groups[2]['lr'] *= args.decay
            # wordenc_opt.param_groups[0]['lr'] *= decay_rate
            # sentenc_opt.param_groups[0]['lr'] *= decay_rate
            # contenc_opt.param_groups[0]['lr'] *= decay_rate
            # cmdec_opt.param_groups[0]['lr'] *= decay_rate
            # dec_opt.param_groups[0]['lr'] *= decay_rate

            # audioenc_opt.param_groups[0]['lr'] *= decay_rate
            # contAudenc_opt.param_groups[0]['lr'] *= decay_rate
            # cmdecAud_opt.param_groups[0]['lr'] *= decay_rate

        if over_fitting >= args.patience:
            break


def comput_class_loss(log_prob, target, weights):
    """ classification loss """
    loss = F.nll_loss(F.log_softmax(log_prob), target.view(
        target.size(0)), weight=weights, reduction='sum')
    loss /= target.size(0)

    return loss


def loss_weight(tr_ladict, ladict, focus_dict, rate=1.0):
    """ loss weights """
    min_emo = float(min([tr_ladict.word2count[w] for w in focus_dict]))
    weight = [math.pow(min_emo / tr_ladict.word2count[k], rate) if k in focus_dict
              else 0 for k, v in ladict.word2count.items()]
    weight = np.array(weight)
    weight /= np.sum(weight)

    return weight


def emoeval(model, data_loader, args):
    """ data_loader only input 'dev' """
    focus_emo = args.focus_emo
    model.eval()
    # wordenc.eval()
    # sentenc.eval()
    # contenc.eval()
    # dec.eval()
    # audioenc.eval()
    # contAudenc.eval()
    # cmdec.eval()
    # cmdecAud.eval()

    # wordenc.cuda()
    # sentenc.cuda()
    # contenc.cuda()
    # cmdec.cuda()
    # dec.cuda()

    # audioenc.cuda()
    # contAudenc.cuda()
    # cmdecAud.cuda()
    # criterion.cuda()

    # weight for loss
    # weight_rate = 0  # eval state without weights
    # if args.dataset in ['MOSI', 'IEMOCAP4v2']:
    #	weight_rate = 0
    # weights = torch.from_numpy(loss_weight(
    #     tr_emodict, emodict, focus_emo, rate=weight_rate)).float()
    feats, feat_attmasks, audio_paths, emo2idx, senti2idx = \
        data_loader['feat'], data_loader['feat_attmask'], \
        data_loader['audio_fea_path'], data_loader['emo2idx'], data_loader['senti2idx']
    if args.dataset in ['MELD', 'IEMOCAP']:
        labels = data_loader['NextEmoLabel']
        label2idx = emo2idx
    elif args.dataset in ['MELD_three', 'IEMOCAP_four']:
        labels = data_loader['NextSentiLabel']
        label2idx = senti2idx

    acc = np.zeros([len(label2idx)], dtype=np.long)  # recall
    num = np.zeros([len(label2idx)], dtype=np.long)  # gold
    # precision, only count those in focus_emo
    preds = np.zeros([len(label2idx)], dtype=np.long)

    focus_idx = [label2idx[emo] for emo in focus_emo]

    # feats, audios, labels = data_loader['feat'], data_loader['audio'], data_loader['label']
    val_loss = 0

    pred_label_list = []
    true_label_list = []
    glob_steps = 0
    for bz in range(len(labels)):
        if feats[bz] == []:
            continue
        feat, lens = Utils.ToTensor(feats[bz], is_len=True)
        feat_attmask, lens_attmasks = Utils.ToTensor(
            feat_attmasks[bz], is_len=True)
        audio, audio_mask = Utils.AudiopathToTensor(
            audio_paths[bz], args=args, is_len=True)
        label = Utils.ToTensor(labels[bz])
        # audio, aud_lens = Utils.ToAudioTensor(audios[bz], is_len=True)
        # label = Utils.ToTensor(labels[bz])
        # feat = Variable(feat)
        # label = Variable(label)

        # if args.gpu != None:
        # 	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # 	# device = torch.device("cuda: 0")
        # 	wordenc.cuda()
        # 	sentenc.cuda()
        # 	contenc.cuda()
        # 	dec.cuda()

        # 	audioenc.cuda()
        # 	contAudenc.cuda()
        if feat.shape[0] == 3:
            feat = torch.cat(
                [torch.zeros(1, feat.shape[1]).long(), feat], 0)
            feat_attmask = torch.cat(
                [torch.zeros(1, feat_attmask.shape[1]).long(), feat_attmask], 0)
            audio = torch.cat(
                [torch.zeros(1, audio.shape[1], audio.shape[2]), audio], 0)
            audio_mask = torch.cat(
                [torch.zeros(1, audio_mask.shape[1]).int(), audio_mask], 0)
        feat = feat.cuda()
        feat_attmask = feat_attmask.cuda()
        audio = audio.cuda()
        audio_mask = audio_mask.cuda()
        label = label.cuda()
        loss, pred_prob = model(
            feat, feat_attmask, audio, audio_mask, label)
        # w_embed = wordenc(feat)
        # s_embed = sentenc(w_embed, lens)[0]

        # audio_scr = audioenc(audio, aud_lens)[0]

        # # for no context
        # sr_text = s_embed[-2] + s_embed[-3]
        # sr_audio = audio_scr[-2] + audio_scr[-3]

        # s_dec_context = sr_text.unsqueeze(0)
        # contAud_scr_dec = sr_audio.unsqueeze(0)

        #log_prob = dec(s_embed)
        # s_context = contenc(s_embed)
        # s_context = s_embed
        # s_dec_context = cmdec(t_context=s_context)
        # # s_concat = torch.cat([s_embed, s_context], dim=-1)

        # audio_scr = audioenc(audio, aud_lens)[0]
        # # contAud_scr = contAudenc(audio_scr)
        # contAud_scr = audio_scr
        # # a_concat = torch.cat([audio_scr,contAud_scr],dim=-1)
        # contAud_scr_dec = cmdecAud(t_context=contAud_scr)

        # log_prob = dec(s_dec_context, contAud_scr_dec)

        log_prob = pred_prob

        # log_prob = dec(s_concat,a_concat)
        # log_prob = dec(s_context)
        # print(log_prob, label)
        # val loss
        # loss = comput_class_loss(log_prob, label, weights)
        # loss = criterion(log_prob, label.squeeze(1))
        val_loss += loss.item()
        glob_steps += 1
        # accuracy

        emo_predidx = torch.argmax(log_prob, dim=1)  # 1x1
        emo_true = label.view(label.size(0))

        pred_label_list.append(emo_predidx[0])
        true_label_list.append(label[0].item())
        for lb in range(emo_true.size(0)):
            idx = emo_true[lb].item()
            num[idx] += 1
            if emo_true[lb] == emo_predidx[lb]:
                acc[idx] += 1
            # count for precision
            # if emo_true[lb] in focus_idx and emo_predidx[lb] in focus_idx:
            preds[emo_predidx[lb]] += 1

    # stop()
    # for t,p in zip(true_label_list,pred_label_list,preds):
    # 	if t == p:
    # 		acc[t] += 1
    # 	num[t] += 1
    # 	preds[p] += 1
    # label_len = emodict.n_words

    # pacc = [acc[i] for i in range(label_len)]
    # pnum = [num[i] for i in range(label_len)]
    # pwACC = sum(pacc) / sum(pnum) * 100
    # ACCs = [np.round(acc[i] / num[i] * 100, 2) if num[i] != 0 else 0 for i in range(label_len)]
    # pACCs = [ACCs[i] for i in range(label_len) ]
    # paACC = sum(pACCs) / len(pACCs)
    # pACCs = ACCs # recall

    # # macro f1
    # TP = acc
    # pPREDs = preds_add
    # pPRECs = [np.round(tp/p*100,2) if p>0 else 0 for tp,p in zip(TP,pPREDs)] # precision
    # pF1s = [np.round(2*r*p/(r+p),2) if r+p>0 else 0 for r,p in zip(pACCs,pPRECs)]
    # F1 = sum(pF1s)/len(pF1s)

    # # micro f1
    # miTP = acc
    # miPREDs =preds_add
    # minum = [num[i] for i in range(label_len)]
    # precison = sum(miTP)/sum(miPREDs) *100
    # recall = sum(miTP) / sum(minum) *100
    # micro_F1 = 2*precison*recall/(precison+recall)

    id2label = {id: label for label, id in label2idx.items()}
    pacc = [acc[i]
            for i in range(len(label2idx)) if id2label[i] in focus_emo]
    pnum = [num[i]
            for i in range(len(label2idx)) if id2label[i] in focus_emo]
    pwACC = sum(pacc) / sum(pnum) * 100
    ACCs = [np.round(acc[i] / num[i] * 100, 2) if num[i] !=
            0 else 0 for i in range(len(label2idx))]
    pACCs = [ACCs[i]
             for i in range(len(label2idx)) if id2label[i] in focus_emo]
    paACC = sum(pACCs) / len(pACCs)
    pACCs = [ACCs[label2idx[w]] for w in focus_emo]  # recall

    # macro f1
    TP = [acc[label2idx[w]] for w in focus_emo]
    pPREDs = [preds[label2idx[w]] for w in focus_emo]
    pPRECs = [np.round(tp/p*100, 2) if p > 0 else 0 for tp,
              p in zip(TP, pPREDs)]  # precision
    pF1s = [np.round(2*r*p/(r+p), 2) if r+p >
            0 else 0 for r, p in zip(pACCs, pPRECs)]
    F1 = sum(pF1s)/len(pF1s)

    # micro f1
    miTP = [acc[label2idx[w]] for w in focus_emo]
    miPREDs = [preds[label2idx[w]] for w in focus_emo]
    minum = [num[i]
             for i in range(len(label2idx)) if id2label[i] in focus_emo]
    precison = sum(miTP)/sum(miPREDs) * 100
    recall = sum(miTP) / sum(minum) * 100
    micro_F1 = 2*precison*recall/(precison+recall)

    # Total = [pACCs] + [pF1s] + [np.round(pwACC,2), np.round(paACC,2), np.round(F1,2),np.round(precison,2),np.round(recall,2),np.round(micro_F1,2), -val_loss]
    Total = [pACCs] + [pF1s] + [np.round(pwACC, 2), np.round(paACC, 2), np.round(F1, 2), np.round(
        precison, 2), np.round(recall, 2), np.round(micro_F1, 2), val_loss/glob_steps]

    model.train()
    return Total


# for case study
def Casestudy(wordenc, sentenc, contenc, cmdec, dec, audioenc, contAudenc, cmdecAud, data_loader, emodict, args):
    wordenc.eval()
    sentenc.eval()
    contenc.eval()
    dec.eval()
    audioenc.eval()
    contAudenc.eval()
    cmdec.eval()
    cmdecAud.eval()

    wordenc.cuda()
    sentenc.cuda()
    contenc.cuda()
    cmdec.cuda()
    dec.cuda()

    audioenc.cuda()
    contAudenc.cuda()
    cmdecAud.cuda()
    # criterion.cuda()


# Ses05F_impro04_F005
# Ses05F_script01_1_F002
# Ses05F_impro07_M014

    feats, audios, labels, textes = data_loader['feat'], data_loader[
        'audio'], data_loader['label'], data_loader['text']
    # feats, labels = data_loader['feat'], data_loader['label']
    label_preds = []

    for bz in range(len(labels)):
        feat, lens = Utils.ToTensor(feats[bz], is_len=True)
        audio, aud_lens = Utils.ToAudioTensor(audios[bz], is_len=True)
        label = Utils.ToTensor(labels[bz])
        feat = Variable(feat)
        label = Variable(label)
        text = textes[bz]
        # if args.gpu != None:
        # 	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # 	device = torch.device("cuda: 0")
        # 	wordenc.cuda(device)
        # 	sentenc.cuda(device)
        # 	contenc.cuda(device)
        # 	dec.cuda(device)
        # 	feat = feat.cuda(device)
        # 	label = label.cuda(device)
        feat = feat.cuda()
        audio = audio.cuda()
        label = label.cuda()

        w_embed = wordenc(feat)
        s_embed = sentenc(w_embed, lens)[0]
        # s_context = contenc(s_embed)
        s_context = s_embed
        s_dec_context = cmdec(t_context=s_context)
        # s_concat = torch.cat([s_embed, s_context], dim=-1)
        # log_prob = dec(s_concat)
        audio_scr = audioenc(audio, aud_lens)[0]
        contAud_scr = audio_scr
        contAud_scr_dec = cmdecAud(t_context=contAud_scr)
        log_prob = dec(s_dec_context, contAud_scr_dec)
        log_prob = log_prob[-1].unsqueeze(0)

        emo_pred = torch.argmax(log_prob, dim=1)
        emo_true = label.view(label.size(0))

        label_pred = []
        for lb in range(emo_true.size(0)):
            true_idx = emo_true[lb].item()
            pred_idx = emo_pred[lb].item()
            label_pred.append(
                (emodict.index2word[true_idx], emodict.index2word[pred_idx]))
        label_preds.append({str(bz): {'label': label_pred, 'text': text}})

        path = '{}_finetune?{}_case.json'.format(
            args.dataset, str(args.load_model))
        Utils.saveToJson(path, label_preds)

    return 1
