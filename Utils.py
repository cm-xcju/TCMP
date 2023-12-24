"""
Utils.

"""
import json
import pickle
from pickletools import optimize
import joblib
import torch
import os
import math
import random
import numpy as np
import Const
import time
from pdb import set_trace as stop
import torch.optim as optim
# Timer


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def ToTensor(list, is_len=False):
    np_ts = np.array(list)
    tensor = torch.from_numpy(np_ts).long()

    if is_len:
        mat1 = np.equal(np_ts, Const.PAD)
        mat2 = np.equal(mat1, False)
        lens = np.sum(mat2, axis=1)
        return tensor, lens

    return tensor


def ToAudioTensor(audio, is_len=False):
    tensor = torch.tensor(audio).float()
    pad_tensor = torch.zeros(tensor.shape[1])
    if is_len:
        # mat1 = np.equal(audio,0)
        mat1 = np.sum(audio, 2)
        mat2 = np.equal(mat1, 0)
        mat3 = np.equal(mat2, False)
        lens = np.sum(mat3, axis=1)
        lens = np.array([le if le > 0 else 1 for le in lens.tolist()])
        return tensor, lens

    return tensor


def AudiopathToTensor(audio_paths, args, is_len=False):

    audio_feas = []
    input_values_all = []
    attention_masks_all = []
    savepath = args.savepath
    max_hubert_vec_maxsize = args.max_hubert_vec_maxsize
    zero_mask_pad_tensor = torch.zeros(1, max_hubert_vec_maxsize)
    zero_input_pad_tensor = torch.zeros(1, max_hubert_vec_maxsize, 2)
    # movie_vectors/friends.s01e05/seg363.job
    for audio_path in audio_paths:
        fea_path = os.path.join(savepath, audio_path)
        audio_fea = joblib.load(fea_path)
        if audio_fea['input_values'].shape[-1] != 2:
            audio_fea['input_values'] = audio_fea['input_values'].unsqueeze(
                -1).repeat(1, 1, 2)
        if is_len:

            input_values = audio_fea.input_values
            attention_masks = audio_fea.attention_mask
            # pad
            if input_values.shape[1] > max_hubert_vec_maxsize:
                input_values = input_values[:, :max_hubert_vec_maxsize]
                attention_masks = attention_masks[:, :max_hubert_vec_maxsize]
            elif input_values.shape[1] < max_hubert_vec_maxsize:
                input_values = torch.cat(
                    [input_values, zero_input_pad_tensor[:, :max_hubert_vec_maxsize-input_values.shape[1]]], 1)
                attention_masks = torch.cat(
                    [attention_masks, zero_mask_pad_tensor[:, :max_hubert_vec_maxsize-attention_masks.shape[1]]], 1)
            input_values_all.append(input_values)
            attention_masks_all.append(attention_masks)
        audio_feas.append(audio_fea)

    if is_len:
        return torch.cat(input_values_all, 0), torch.cat(attention_masks_all, 0)

    return audio_feas


def scrmodel_saver(model, path, module, dataset):
    if not os.path.isdir(path):
        os.makedirs(path)
    model_path = '{}/{}_{}.pt'.format(path, module, dataset)
    torch.save(model, model_path)


def scrmodel_saver_new(model, path):
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    # model_path = './{}'.format(path)

    torch.save(model, path)

# model saver


def revmodel_saver(model, path, module, dataset, finetune):
    if not os.path.isdir(path):
        os.makedirs(path)
    model_path = '{}/{}_{}_finetune?{}.pt'.format(
        path, module, dataset, str(finetune))
    torch.save(model, model_path)

# model loader


def revmodel_loader(path, module, dataset, finetune):
    model_path = '{}/{}_{}_finetune?{}.pt'.format(
        path, module, dataset, str(finetune))
    model = torch.load(model_path, map_location='cpu')
    return model


def saveToJson(path, object):
    t = json.dumps(object, indent=4)
    f = open(path, 'w')
    f.write(t)
    f.close()

    return 1


def saveToPickle(path, object):
    file = open(path, 'wb')
    pickle.dump(object, file)
    file.close()

    return 1


def loadFrPickle(path):

    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj


def loadFrPickle(path):

    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj


def saveToJoblib(path, object):
    file = open(path, 'wb')
    joblib.dump(object, file)
    file.close()

    print("joblib file save success")


def loadFrJoblib(path):

    # file = open(path, 'rb')
    obj = joblib.load(path)
    # file.close()
    return obj


def load_bin_vec(filename, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    dtype: word2vec float32, glove float64;
    Word2vec's input is encoded in UTF-8, but output is encoded in ISO-8859-1
    """
    print('Initilaize with Word2vec 300d word vectors!')
    word_vecs = {}
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split()[0:2])
        binary_len = np.dtype('float32').itemsize * layer1_size
        num_tobe_assigned = 0
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('iso-8859-1')
                if ch == ' ':
                    word = ''.join(word)
                    # print(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                vector = np.fromstring(f.read(binary_len), dtype='float32')
                word_vecs[word] = vector / np.sqrt(sum(vector**2))
                num_tobe_assigned += 1
            else:
                f.read(binary_len)
        print("Found words {} in {}".format(vocab_size, filename))
        match_rate = round(num_tobe_assigned/len(vocab)*100, 2)
        print("Matched words {}, matching rate {} %".format(
            num_tobe_assigned, match_rate))
    return word_vecs


def load_txt_glove(filename, vocab):
    """
    Loads 300x1 word vecs from Glove
    dtype: glove float64;
    """
    print('Initilaize with Glove 300d word vectors!')
    word_vecs = {}
    vector_size = 300

    with open(filename, "r") as f:
        vocab_size = 0
        num_tobe_assigned = 0
        for line in f:
            vocab_size += 1
            splitline = line.split()
            word = " ".join(splitline[0:len(splitline) - vector_size])
            if word in vocab:
                vector = np.array([float(val)
                                  for val in splitline[-vector_size:]])
                word_vecs[word] = vector / np.sqrt(sum(vector**2))
                num_tobe_assigned += 1

        print("Found words {} in {}".format(vocab_size, filename))
        match_rate = round(num_tobe_assigned/len(vocab)*100, 2)
        print("Matched words {}, matching rate {} %".format(
            num_tobe_assigned, match_rate))
    return word_vecs


def load_pretrain(d_word_vec, diadict, type='word2vec'):
    """ initialize nn.Embedding with pretrained """
    if type == 'word2vec':
        filename = 'word2vec300.bin'
        word2vec = load_bin_vec(filename, diadict.word2index)
    elif type == 'glove':
        filename = '/data/xcju/code/project_6/dataset/preEmotion/glove300.txt'
        word2vec = load_txt_glove(filename, diadict.word2index)

    # initialize a numpy tensor
    embedding = np.random.uniform(-0.01, 0.01, (diadict.n_words, d_word_vec))
    for w, v in word2vec.items():
        embedding[diadict.word2index[w]] = v

    # zero padding
    embedding[Const.PAD] = np.zeros(d_word_vec)

    return embedding


def load_char_vec(filename, vocab):
    """
    Loads 300x1 char vecs from glove.840B.300d-char.txt
    dtype: glove float64;
    UTF-8, but output is encoded in ISO-8859-1
    """
    char_vecs = {}
    with open(filename, "r") as f:
        vocab_size = 94
        layer_size = 300
        num_tobe_assigned = 0
        for line in f:
            splitline = line.split()
            char = splitline[0]
            if char in vocab:
                vector = np.array([float(v) for v in splitline[1:]])
                assert len(vector) == layer_size
                char_vecs[char] = vector / np.sqrt(sum(vector**2))
                num_tobe_assigned += 1
        print("Found chars {} in {}".format(vocab_size, filename))
        match_rate = round(num_tobe_assigned/len(vocab)*100, 2)
        print("Matched chars {}, matching rate {} %".format(
            num_tobe_assigned, match_rate))
    return char_vecs


def load_charvecs(d_char_vec, n_chars, char2idx):
    """ initialize nn.Embedding with pretrained """
    filename = 'glove.840B.300d-char.txt'
    char2vec = load_char_vec(filename, char2idx)

    # initialize a numpy tensor
    embedding = np.random.uniform(-0.01, 0.01, (n_chars, d_char_vec))
    for c, v in char2vec.items():
        embedding[char2idx[c]] = v

    # zero padding
    embedding[Const.cPAD] = np.zeros(d_char_vec)

    return embedding


def shuffle_lists(featllist, labellist=None, thirdparty=None, fourthparty=None, fivethparty=None):

    if labellist == None:
        random.shuffle(featllist)
        return featllist
    elif labellist != None and thirdparty == None:
        combined = list(zip(featllist, labellist))
        random.shuffle(combined)
        featllist, labellist = zip(*combined)
        return featllist, labellist
    elif thirdparty != None and fourthparty == None:
        combined = list(zip(featllist, labellist, thirdparty))
        random.shuffle(combined)
        featllist, labellist, thirdparty = zip(*combined)
        return featllist, labellist, thirdparty
    elif fourthparty != None and fivethparty == None:
        combined = list(zip(featllist, labellist, thirdparty, fourthparty))
        random.shuffle(combined)
        featllist, labellist, thirdparty, fourthparty = zip(*combined)
        return featllist, labellist, thirdparty, fourthparty
    else:
        combined = list(zip(featllist, labellist, thirdparty,
                        fourthparty, fivethparty))
        random.shuffle(combined)
        featllist, labellist, thirdparty, fourthparty, fivethparty = zip(
            *combined)
        return featllist, labellist, thirdparty, fourthparty, fivethparty


def param_clip(model, optimizer, batch_size, max_norm=10):
    # gradient clipping
    shrink_factor = 1
    total_norm = 0

    for p in model.parameters():
        if p.requires_grad:
            p.grad.data.div_(batch_size)
            total_norm += p.grad.data.norm() ** 2
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        # print("Total norm of grads {}".format(total_norm))
        shrink_factor = max_norm / total_norm
    current_lr = optimizer.param_groups[0]['lr']

    return current_lr, shrink_factor


def get_new_dialogs(scripts, script_attmasks, audios, negs, neg_attmasks, negaudios, winstride, window, labels=None, warm=None):

    new_scripts = []
    new_scripts_attmasks = []
    new_audios = []
    new_negs = []
    new_negs_attmasks = []
    new_negaudios = []
    new_labels = []
    if warm == None:
        warm = 0

    for bz in range(len(scripts)):
        # strides = len(scripts[bz]) // winstride
        for i in range(warm, len(scripts[bz]), winstride):  # winstride=window
            begin = i  # if i-window < 0 else i - window
            end = len(scripts[bz]) if i + \
                window >= len(scripts[bz]) else i + window
            if end - begin < 3:
                continue

            script = scripts[bz][begin:end]
            script_attmask = script_attmasks[bz][begin:end]
            audio = audios[bz][begin:end]
            neg = negs[bz]
            neg_attmask = neg_attmasks[bz]
            negaudio = negaudios[bz]
            # label = labels[bz][begin:end-2]

            new_scripts.append(script)
            new_scripts_attmasks.append(script_attmask)
            new_audios.append(audio)
            new_negs.append(neg)
            new_negs_attmasks.append(neg_attmask)
            new_negaudios.append(negaudio)
            # new_labels.append(label)

    return new_scripts, new_scripts_attmasks, new_audios, new_negs, new_negs_attmasks, new_negaudios  # ,new_labels


def neg_sample(scripts, scr_idx, num_neg=10, window=None):
    set_len = len(scripts)
    conv_len = len(scripts[scr_idx])
    to_be_avoid = []
    to_be_avoid.append(scr_idx)
    if window != None:
        for i in range(10):
            if scr_idx+i < len(scripts):
                to_be_avoid.append(scr_idx+i)
            if scr_idx-i > 0:
                to_be_avoid.append(scr_idx-i)

    # produce negative samples
    neg = []
    for j in range(num_neg):
        rd1 = random.randrange(0, set_len)
        while rd1 in to_be_avoid:
            rd1 = random.randrange(0, set_len)
        to_be_avoid.append(rd1)
        scr_samp = scripts[rd1]
        num_utt = len(scr_samp)
        rd2 = random.randrange(0, num_utt)
        neg.append(scr_samp[rd2])

    # produce label
    la_idxs = [1] + [0] * num_neg
    laidxs = [la_idxs] * (conv_len - 2)
    return neg, laidxs


def neg_pair_sample(scripts, script_attmasks, audio_paths, scr_idx, args, window=None):
    num_neg = args.num_neg
    set_len = len(scripts)
    conv_len = len(scripts[scr_idx])
    to_be_avoid = []
    to_be_avoid.append(scr_idx)
    if window != None:
        for i in range(10):
            if scr_idx+i < len(scripts):
                to_be_avoid.append(scr_idx+i)
            if scr_idx-i > 0:
                to_be_avoid.append(scr_idx-i)

    # produce negative samples
    negscr = []
    negscr_mask = []
    neg_audio_paths = []

    for j in range(num_neg):
        rd1 = random.randrange(0, set_len)
        while rd1 in to_be_avoid:
            rd1 = random.randrange(0, set_len)
        to_be_avoid.append(rd1)
        scr_samp = scripts[rd1]
        scr_mask_samp = script_attmasks[rd1]
        audio_paths_samp = audio_paths[rd1]
        num_utt = len(scr_samp)
        rd2 = random.randrange(0, num_utt)
        negscr.append(scr_samp[rd2])
        negscr_mask.append(scr_mask_samp[rd2])
        neg_audio_paths.append(audio_paths_samp[rd2])
    negscr = ToTensor(negscr)
    negscr_mask = ToTensor(negscr_mask)
    neg_audio, neg_audio_mask = AudiopathToTensor(
        neg_audio_paths, args=args, is_len=True)
    # produce label
    # la_idxs = [1] + [0] * num_neg
    # laidxs = [la_idxs] * (conv_len - 2)
    # , laidxs
    return negscr, negscr_mask, neg_audio, neg_audio_mask


def loss_weight(data, args):
    dataset = args.dataset
    focus_emo = args.focus_emo
    cont = [0] * len(focus_emo)
    if dataset in ['MELD', 'IEMOCAP']:
        for phase in data.keys():
            da = data[phase]['NextEmoLabel']
            for tt in da:
                id = tt[0][0]
                cont[id] += 1
    if dataset in ['MELD_three', 'IEMOCAP_four']:
        for phase in data.keys():
            da = data[phase]['NextSentiLabel']
            for tt in da:
                id = tt[0][0]
                cont[id] += 1
    sum_cont = float(sum(cont))
    weight = [math.pow(sum_cont/i, 1) for i in cont]
    weight = np.array(weight)
    weight /= np.sum(weight)
    weight *= len(focus_emo)
    return weight


def prepare_optimizer(args, model, num_train_steps):

    #  optim.Adam(model.parameters(), lr=args.lr)
    # if args.fp16:
    #     param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
    #                        for n, param in model.named_parameters()]
    # elif args.optimize_on_cpu:
    #     param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
    #                        for n, param in model.named_parameters()]
    # else:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    return optimizer, param_optimizer


def prepare_optimizer_2(args, model):
    # pretrain_model_params = list(map(id, model.pretrain_model.parameters()))
    roberta_params = list(map(id, model.pre_model.FeaEnc.roberta.parameters()))
    hubert_params = list(map(id, model.pre_model.FeaEnc.hubert.parameters()))
    # bert_params = list(map(id, model.bert.parameters()))
    # id(p) not in resnet152_params and id(p) not in pretrain_model_params and
    base_params = filter(lambda p:
                         id(p) not in hubert_params and id(
                             p) not in roberta_params,
                         # and p.requires_grad,
                         model.parameters())
    # no_decay = ['bias', 'LayerNorm']
    params = [
        {'params': base_params, 'lr': args.lr},
        {'params': model.pre_model.FeaEnc.roberta.parameters(
        ), 'lr': args.learning_rate_pretrained},
        {'params': model.pre_model.FeaEnc.hubert.parameters(),
         'lr': args.learning_rate_pretrained},
    ]
    # paras=[
    #         {'params':}
    #         ]
    # torch.optim.Adam(paras, weight_decay=args.)
    # torch.optim.Adam(paras, weight_decay=args.wdecay)
    # scheduler_rel = torch.optim.lr_scheduler.LambdaLR(optimizer)
    # optimizer = BERTAdam(paras,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=num_train_steps)
    optimizer = optim.AdamW(params)
    return optimizer, params

def prepare_optimizer_for_pretrained(args, model):
    # pretrain_model_params = list(map(id, model.pretrain_model.parameters()))
    roberta_params = list(map(id, model.FeaEnc.roberta.parameters()))
    hubert_params = list(map(id, model.FeaEnc.hubert.parameters()))
    # bert_params = list(map(id, model.bert.parameters()))
    # id(p) not in resnet152_params and id(p) not in pretrain_model_params and
    base_params = filter(lambda p:
                         id(p) not in hubert_params and id(
                             p) not in roberta_params,
                         # and p.requires_grad,
                         model.parameters())
    # no_decay = ['bias', 'LayerNorm']
    params = [
        {'params': base_params, 'lr': args.lr},
        {'params': model.FeaEnc.roberta.parameters(
        ), 'lr': args.learning_rate_pretrained},
        {'params': model.FeaEnc.hubert.parameters(),
         'lr': args.learning_rate_pretrained},
    ]
    # paras=[
    #         {'params':}
    #         ]
    # torch.optim.Adam(paras, weight_decay=args.)
    # torch.optim.Adam(paras, weight_decay=args.wdecay)
    # scheduler_rel = torch.optim.lr_scheduler.LambdaLR(optimizer)
    # optimizer = BERTAdam(paras,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=num_train_steps)
    optimizer = optim.AdamW(params)
    return optimizer, params
