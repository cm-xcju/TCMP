"""
Train on OpSub dataset
"""
from audioop import avg
import os
import time
import numpy as np
import random
import torch
import torch.optim as optim
from torch.autograd import Variable
import Utils
from datetime import datetime
from pdb import set_trace as stop
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter


def lmtrain(model, criterion, optimizer, data_loader, args):
    """
    :data_loader input the whole field
    """
    # start time
    time_st = time.time()
    decay_rate = 0.75

    # dataloaders
    train_loader = data_loader['train']
    dev_loader = data_loader['dev']
    test_loader = data_loader['test']

    scripts_raw, script_attmasks_raw, audio_paths_raw, negs_raw, neg_attmasks_raw, negaudio_paths_raw, labels_raw = train_loader['script'], train_loader['script_attmask'], train_loader[
        'audio_fea_path'], train_loader['neg'], train_loader['neg_attmask'], train_loader['neaudio_fea_path'], train_loader['label']
    # scripts, audios,negs,negaudios, labels = train_loader['script'],train_loader['audio'], train_loader['neg'],train_loader['negaudio'], train_loader['label']

    lr = args.lr
    # wordenc_opt = optim.Adam(wordenc.parameters(), lr=lr)
    # sentenc_opt = optim.Adam(sentenc.parameters(), lr=lr)
    # contenc_opt = optim.Adam(contenc.parameters(), lr=lr)
    # dec_opt = optim.Adam(dec.parameters(), lr=lr)
    # #audio
    # audio_opt = optim.Adam(audioenc.parameters(), lr=lr)
    # contAudenc_opt = optim.Adam(contAudenc.parameters(), lr=lr)
    # decAud_opt = optim.Adam(decAud.parameters(), lr=lr)

    # cmdecA2T_opt = optim.Adam(cmdecA2T.parameters(), lr=lr)
    # cmdecT2A_opt = optim.Adam(cmdecT2A.parameters(), lr=lr)

    criterion.cuda()
    model.cuda()
    model.train()

    # wordenc.cuda()
    # sentenc.cuda()
    # contenc.cuda()
    # dec.cuda()

    # audioenc.cuda()
    # contAudenc.cuda()
    # decAud.cuda()

    # cmdecA2T.cuda()
    # cmdecT2A.cuda()

    # wordenc.train()
    # sentenc.train()
    # contenc.train()
    # dec.train()

    # audioenc.train()
    # contAudenc.train()
    # decAud.train()

    # cmdecA2T.train()
    # cmdecT2A.train()

    over_fitting = 0
    cur_best = 0
    glob_steps = 0
    report_loss = 0
    loss_minbatch = 0

    for epoch in range(1, args.epochs + 1):

        time_epoch = time.time()
        warm = epoch % args.window
        scripts_raw, script_attmasks_raw, audio_paths_raw, negs_raw, negaudio_paths_raw = Utils.shuffle_lists(
            scripts_raw, script_attmasks_raw, audio_paths_raw, negs_raw, negaudio_paths_raw)
        print("===========Epoch:{}==============".format(epoch))
        print("-{}-{}".format(epoch, datetime.now()))

        window = args.window
        winstride = args.winstride
        scripts, script_attmasks, audio_paths, negs, neg_attmasks, negaudio_paths = Utils.get_new_dialogs(
            scripts_raw, script_attmasks_raw, audio_paths_raw, negs_raw, neg_attmasks_raw, negaudio_paths_raw, winstride, window, warm=warm)
        # scripts, audios, negs, negaudios, labels = Utils.get_new_dialogs(
        #     scripts, audios, negs, negaudios, labels, winstride, window)
        print('='*10, 'scripts length', '='*10, len(scripts), '='*20)
        # for bz in tqdm(range(len(scripts)), mininterval=0.5, desc='(Training)', leave=False):
        for bz in range(len(scripts)):
            # if bz > 5:
            #     continue
            # for bz in range(len(scripts)):
            time_step = time.time()
            # begin = (bz // window) * window
            # end = （bz // window + 1）* window
            # if end > :
            # 	script_wn = scripts[bz][begin:end]

            # tensorize a dialog list

            script, lens = Utils.ToTensor(scripts[bz], is_len=True)
            script_attmask, lens_attmasks = Utils.ToTensor(
                script_attmasks[bz], is_len=True)

            # audio, aud_lens = Utils.ToAudioTensor(audios[bz], is_len=True)
            audio, audio_mask = Utils.AudiopathToTensor(
                audio_paths[bz], args=args, is_len=True)

            # negative sampling
            # if window != None:
            # neg_sampled = Utils.neg_sample(
            #     scripts, bz, num_neg=10, window=window)
            # neg_audio_sampled, = Utils.neg_sample(
            #     audio_paths, bz, num_neg=10, window=window)

            neg_scr, neg_scr_mask, neg_audio, neg_audio_mask = Utils.neg_pair_sample(
                scripts, script_attmasks, audio_paths, bz, args=args, window=window)
            # else:
            #     neg_sampled = neg_sample(
            #         scripts, bz, num_neg=10)
            #     neg_audio_sampled= neg_sample(
            #         audio_paths, bz, num_neg=10)

            # neg, lenn = Utils.ToTensor(neg_sampled, is_len=True)
            # negaudio, audlenn = Utils.ToAudioTensor(
            #     neg_audio_sampled, is_len=True)

            # label = Utils.ToTensor(label_sampled)
            # script = Variable(script)
            # neg = Variable(neg)
            # label = Variable(label).float()

            # if args.gpu != None:
            # 	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            # device = torch.device("cuda: 0")
            # wordenc.cuda()
            # sentenc.cuda()
            # contenc.cuda()
            # dec.cuda()

            # audioenc.cuda()
            # contAudenc.cuda()
            # decAud.cuda()

            # criterion.cuda()
            script = script.cuda()
            script_attmask = script_attmask.cuda()
            audio = audio.cuda()
            audio_mask = audio_mask.cuda()
            neg_scr = neg_scr.cuda()
            neg_scr_mask = neg_scr_mask.cuda()
            neg_audio = neg_audio.cuda()
            neg_audio_mask = neg_audio_mask.cuda()
            # label = label.cuda()

            # for window slide
            loss, pred_t, pred_a, pred_mm = model(
                script, script_attmask, audio, audio_mask, neg_scr, neg_scr_mask, neg_audio, neg_audio_mask)

            # word_scr = wordenc(script)
            # sent_scr = sentenc(word_scr, lens)[0]
            # word_neg = wordenc(neg)
            # sent_neg = sentenc(word_neg, lenn)[0]
            # # cont_scr = contenc(sent_scr)
            # cont_scr = sent_scr
            # prob0 = dec(sent_scr, cont_scr, sent_neg)

            # print(log_prob, label)
            # loss_sent = criterion(prob0.view(-1), label.view(-1))

            # # audio_scr = audioenc(audio, lens)[0]
            # # word_neg = wordenc(neg)

            # audio_scr = audioenc(audio, aud_lens)[0]
            # # contAud_scr = contAudenc(audio_scr)
            # contAud_scr = audio_scr
            # audio_neg = audioenc(negaudio, audlenn)[0]
            # prob1 = decAud(audio_scr, contAud_scr, audio_neg)
            # # print(log_prob, label)

            # loss_audio = criterion(prob1.view(-1), label.view(-1))
            # for cross modal

            # prob0_a = cmdecA2T(audio_scr, sent_scr, audio_neg)
            # prob1_a = cmdecT2A(sent_scr, audio_scr, sent_neg)

            # loss_sent_a = criterion(prob0_a.view(-1), label.view(-1))
            # loss_audio_a = criterion(prob1_a.view(-1), label.view(-1))

            # loss = loss_sent + loss_audio + loss_sent_a + loss_audio_a

            loss.backward()

            report_loss += loss.item()

            loss_minbatch += loss.item()
            glob_steps += 1

            # gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(sentenc.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(contenc.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=5)

            # torch.nn.utils.clip_grad_norm_(audioenc.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(contAudenc.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(decAud.parameters(), max_norm=5)

            # torch.nn.utils.clip_grad_norm_(cmdecT2A.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(cmdecA2T.parameters(), max_norm=5)

            # wordenc_opt.step()
            # sentenc_opt.step()
            # contenc_opt.step()
            # dec_opt.step()

            # audio_opt.step()
            # contAudenc_opt.step()
            # decAud_opt.step()

            # cmdecT2A_opt.step()
            # cmdecA2T_opt.step()

            # wordenc_opt.zero_grad()
            # sentenc_opt.zero_grad()
            # contenc_opt.zero_grad()
            # dec_opt.zero_grad()

            # audio_opt.zero_grad()
            # contAudenc_opt.zero_grad()
            # decAud_opt.zero_grad()

            # cmdecT2A_opt.zero_grad()
            # cmdecA2T_opt.zero_grad()
            if bz % args.accum_num == 0:
                optimizer.step()
                optimizer.zero_grad()

            if glob_steps % args.report_loss == 0:
                print("{} Steps: {} Loss: {} LR: {}".format(datetime.now(
                ), glob_steps, report_loss/args.report_loss, optimizer.param_groups[0]['lr']))
                report_loss = 0
            # print('Time step {} \n'.format(Utils.timeSince(time_step)))
        optimizer.step()
        optimizer.zero_grad()
        print('Time epoch {} \n'.format(Utils.timeSince(time_epoch)))
        # avg_loss = report_loss/len(scripts)
        # print('train loss : {} '.format(avg_loss))
        # validate
        with torch.no_grad():
            topkns = lmeval(model, dev_loader, args)
        print("Time {} Validate Text: R1@5 R2@5 R1@11 R2@11 {}".format(
            Utils.timeSince(time_st), topkns[0]))
        print("Time {} Validate Audio: R1@5 R2@5 R1@11 R2@11 {}".format(
            Utils.timeSince(time_st), topkns[1]))
        print("Time {} Validate TextAndaudio: R1@5 R2@5 R1@11 R2@11 {}".format(
            Utils.timeSince(time_st), topkns[2]))
        # print("Time {} Validate Audio2sent: R1@5 R2@5 R1@11 R2@11 {}".format(
        #     Utils.timeSince(time_st), topkns[3]))
        with open(args.train_score_file, 'a') as ft:
            ft.write('Time {} Validate Text:'+str(topkns[0]))
            ft.write('Time {} Validate Audio:'+str(topkns[1]))
            ft.write('Time {} Validate TextAndaudio:'+str(topkns[2]))

        last_best = topkns[0][2]

        if last_best > cur_best:
            # Utils.scrmodel_saver(wordenc, args.save_dir, 'wordenc' +
            #                      str(args.d_hidden_up)+'_'+str(args.d_hidden_low), args.dataset)
            # Utils.scrmodel_saver(sentenc, args.save_dir, 'sentenc' +
            #                      str(args.d_hidden_up)+'_'+str(args.d_hidden_low), args.dataset)
            # Utils.scrmodel_saver(contenc, args.save_dir, 'contenc' +
            #                      str(args.d_hidden_up)+'_'+str(args.d_hidden_low), args.dataset)
            # Utils.scrmodel_saver(
            #     dec, args.save_dir, 'cmdec'+str(args.d_hidden_up)+'_'+str(args.d_hidden_low), args.dataset)

            # Utils.scrmodel_saver(audioenc, args.save_dir, 'audioenc' +
            #                      str(args.d_hidden_up)+'_'+str(args.d_hidden_low), args.dataset)
            # Utils.scrmodel_saver(contAudenc, args.save_dir, 'contAudenc' +
            #                      str(args.d_hidden_up)+'_'+str(args.d_hidden_low), args.dataset)
            # Utils.scrmodel_saver(decAud, args.save_dir, 'decAud' +
            #                      str(args.d_hidden_up)+'_'+str(args.d_hidden_low), args.dataset)

            # Utils.scrmodel_saver(cmdecA2T, args.save_dir, 'cmdecA2T' +
            #                      str(args.d_hidden_up)+'_'+str(args.d_hidden_low), args.dataset)
            # Utils.scrmodel_saver(cmdecT2A, args.save_dir, 'cmdecT2A' +
            #                      str(args.d_hidden_up)+'_'+str(args.d_hidden_low), args.dataset)

            Utils.scrmodel_saver_new(model, args.movie_checkpoint)

            cur_best = last_best
            over_fitting = 0
        else:
            over_fitting += 1
            optimizer.param_groups[0]['lr'] *= decay_rate
            # wordenc_opt.param_groups[0]['lr'] *= decay_rate
            # sentenc_opt.param_groups[0]['lr'] *= decay_rate
            # contenc_opt.param_groups[0]['lr'] *= decay_rate
            # dec_opt.param_groups[0]['lr'] *= decay_rate

            # audio_opt.param_groups[0]['lr'] *= decay_rate
            # contAudenc_opt.param_groups[0]['lr'] *= decay_rate
            # decAud_opt.param_groups[0]['lr'] *= decay_rate

            # cmdecT2A_opt.param_groups[0]['lr'] *= decay_rate
            # cmdecA2T_opt.param_groups[0]['lr'] *= decay_rate

        if over_fitting >= args.patience:
            break


def topkn(matrix, k, n, true_idx=0):
    """
    :param matrix: batch x N, k <= n
    :return:
    """
    batch = matrix.size()[0]
    topk = matrix[:, :n].topk(k, dim=1)
    topk_sum = torch.sum(topk[1].eq(true_idx))

    return topk_sum, batch


def lmeval(model, data_loader, args):
    # def lmeval(wordenc, sentenc, audioenc, contenc, contAudenc, dec, decAud, cmdecA2T, cmdecT2A, data_loader, args):
    """ data_loader only input 'dev' """
    # """ data_loader only input 'dev' """
    model.eval()
    # wordenc.eval()
    # sentenc.eval()
    # audioenc.eval()
    # contenc.eval()

    # contAudenc.eval()
    # dec.eval()
    # decAud.eval()

    # cmdecA2T.eval()
    # cmdecT2A.eval()

    # wordenc.cuda()
    # sentenc.cuda()
    # contenc.cuda()
    # dec.cuda()

    # audioenc.cuda()
    # contAudenc.cuda()
    # decAud.cuda()

    # cmdecA2T.cuda()
    # cmdecT2A.cuda()

    scripts_raw, script_attmasks_raw, audio_paths_raw, negs_raw, neg_attmasks_raw, negaudio_paths_raw, labels_raw = data_loader['script'], data_loader['script_attmask'], data_loader[
        'audio_fea_path'], data_loader['neg'], data_loader['neg_attmask'], data_loader['neaudio_fea_path'], data_loader['label']
    winstride = args.winstride
    window = args.window
    scripts, script_attmasks, audio_paths, negs, neg_attmasks, negaudio_paths = Utils.get_new_dialogs(
        scripts_raw, script_attmasks_raw, audio_paths_raw, negs_raw, neg_attmasks_raw, negaudio_paths_raw, winstride, window)
    top15_all = 0
    top25_all = 0
    top111_all = 0
    top211_all = 0

    top15_aud_all = 0
    top25_aud_all = 0
    top111_aud_all = 0
    top211_aud_all = 0

    top15_t2a_all = 0
    top25_t2a_all = 0
    top111_t2a_all = 0
    top211_t2a_all = 0

    top15_a2t_all = 0
    top25_a2t_all = 0
    top111_a2t_all = 0
    top211_a2t_all = 0

    batch_all = 0
    for bz in range(len(scripts)):
        # tensorize a dialog list
        script, lens = Utils.ToTensor(scripts[bz], is_len=True)
        script_attmask, lens_attmasks = Utils.ToTensor(
            script_attmasks[bz], is_len=True)
        audio, audio_mask = Utils.AudiopathToTensor(
            audio_paths[bz], args=args, is_len=True)
        # audio, aud_lens = Utils.ToAudioTensor(audios[bz], is_len=True)
        neg_scr, lens_neg = Utils.ToTensor(negs[bz], is_len=True)

        neg_scr_mask, lens_attmasks = Utils.ToTensor(
            neg_attmasks[bz], is_len=True)
        neg_audio, neg_audio_mask = Utils.AudiopathToTensor(
            negaudio_paths[bz], args=args, is_len=True)

        # negaudio, Audlenn = Utils.ToAudioTensor(negaudios[bz], is_len=True)
        # label = Utils.ToTensor(labels[bz])
        # script = Variable(script)
        # neg = Variable(neg)
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
        # 	decAud.cuda()
        script = script.cuda()
        script_attmask = script_attmask.cuda()
        audio = audio.cuda()
        audio_mask = audio_mask.cuda()
        neg_scr = neg_scr.cuda()
        neg_scr_mask = neg_scr_mask.cuda()
        neg_audio = neg_audio.cuda()
        neg_audio_mask = neg_audio_mask.cuda()
        # script = script.cuda()
        # label = label.cuda()
        # neg = neg.cuda()
        # audio = audio.cuda()
        # negaudio = negaudio.cuda()

        # word_scr = wordenc(script)
        # sent_scr = sentenc(word_scr, lens)[0]
        # word_neg = wordenc(neg)
        # sent_neg = sentenc(word_neg, lenn)[0]
        # # cont_scr = contenc(sent_scr)
        # cont_scr = sent_scr
        # prob0 = dec(sent_scr, cont_scr, sent_neg)
        # # L-2 x (1+N)
        # # n, k < n
        # audio_scr = audioenc(audio, aud_lens)[0]
        # # contAud_scr = contAudenc(audio_scr)
        # contAud_scr = audio_scr
        # audio_neg = audioenc(negaudio, Audlenn)[0]
        # prob1 = decAud(audio_scr, contAud_scr, audio_neg)

        # prob0_a = cmdecA2T(audio_scr, sent_scr, audio_neg)
        # prob1_a = cmdecT2A(sent_scr, audio_scr, sent_neg)
        loss, pred_t, pred_a, pred_mm = model(
            script, script_attmask, audio, audio_mask, neg_scr, neg_scr_mask, neg_audio, neg_audio_mask)

        prob0_ = torch.sigmoid(pred_t)
        top15, batch = topkn(prob0_, 1, 5)
        top25 = topkn(prob0_, 2, 5)[0]
        top111 = topkn(prob0_, 1, 11)[0]
        top211 = topkn(prob0_, 2, 11)[0]
        top15_all += top15.item()
        top25_all += top25.item()
        top111_all += top111.item()
        top211_all += top211.item()
        batch_all += batch

        prob1_ = torch.sigmoid(pred_a)
        top15, batch = topkn(prob1_, 1, 5)
        top25 = topkn(prob1_, 2, 5)[0]
        top111 = topkn(prob1_, 1, 11)[0]
        top211 = topkn(prob1_, 2, 11)[0]
        top15_aud_all += top15.item()
        top25_aud_all += top25.item()
        top111_aud_all += top111.item()
        top211_aud_all += top211.item()

        prob0_a_ = torch.sigmoid(pred_mm)
        top15, batch = topkn(prob0_a_, 1, 5)
        top25 = topkn(prob0_a_, 2, 5)[0]
        top111 = topkn(prob0_a_, 1, 11)[0]
        top211 = topkn(prob0_a_, 2, 11)[0]
        top15_a2t_all += top15.item()
        top25_a2t_all += top25.item()
        top111_a2t_all += top111.item()
        top211_a2t_all += top211.item()

        # prob1_a_ = torch.sigmoid(prob1_a)
        # top15, batch = topkn(prob1_a_, 1, 5)
        # top25 = topkn(prob1_a_, 2, 5)[0]
        # top111 = topkn(prob1_a_, 1, 11)[0]
        # top211 = topkn(prob1_a_, 2, 11)[0]
        # top15_t2a_all += top15.item()
        # top25_t2a_all += top25.item()
        # top111_t2a_all += top111.item()
        # top211_t2a_all += top211.item()

    topkns0 = [round(float(top15_all)/batch_all, 4),
               round(float(top25_all)/batch_all, 4),
               round(float(top111_all)/batch_all, 4),
               round(float(top211_all)/batch_all, 4)]

    topkns1 = [round(float(top15_aud_all)/batch_all, 4),
               round(float(top25_aud_all)/batch_all, 4),
               round(float(top111_aud_all)/batch_all, 4),
               round(float(top211_aud_all)/batch_all, 4)]

    topkns2 = [round(float(top15_a2t_all)/batch_all, 4),
               round(float(top25_a2t_all)/batch_all, 4),
               round(float(top111_a2t_all)/batch_all, 4),
               round(float(top211_a2t_all)/batch_all, 4)]

    # topkns3 = [round(float(top15_t2a_all)/batch_all, 4),
    #            round(float(top25_t2a_all)/batch_all, 4),
    #            round(float(top111_t2a_all)/batch_all, 4),
    #            round(float(top211_t2a_all)/batch_all, 4)]

    topkns = [topkns0, topkns1, topkns2]  # , topkns3]

    # wordenc.train()
    # sentenc.train()
    # contenc.train()
    # dec.train()

    # audioenc.train()
    # contAudenc.train()
    # decAud.train()

    # cmdecA2T.train()
    # cmdecT2A.train()
    model.train()

    return topkns
