"""
Main function for pre-training on the Conversation Completion task.
Date: 2020/09/24
"""
import numpy as np
import os

import torch
import torch.nn as nn
import Utils
import Const
from Preprocess import Dictionary  # import the object for pickle loading
import Modules
from LMTrain import lmtrain, lmeval
from datetime import datetime
from pdb import set_trace as stop
import random
from models.main_model import TCMP_pre
from config_args import config_pretrain_args, get_pretrain_args
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():

    args = config_pretrain_args()

    # seed
    setup_seed(args.seed)

    # load vocabs
    # print("Loading vocabulary...")
    # glob_vocab = Utils.loadFrPickle(args.vocab_path)
    # # load field
    print("Loading field...")

    # field = Utils.loadFrPickle(args.data_path)
    # field = Utils.loadFrJoblib(args.data_path)
    field = torch.load(args.data_path)

    test_loader = field['test']

    # word embedding
    print("Initializing word embeddings...")

    # embedding = nn.Embedding(
    #     glob_vocab.n_words, args.d_word_vec, padding_idx=Const.PAD)
    # if args.d_word_vec == 300:
    #     if args.embedding != None and os.path.isfile(args.embedding):
    #         np_embedding = Utils.loadFrPickle(args.embedding)
    #     else:
    #         np_embedding = Utils.load_pretrain(
    #             args.d_word_vec, glob_vocab, type='glove')
    #         Utils.saveToPickle(
    #             "../../dataset/preEmotion/embedding.pt", np_embedding)
    #     embedding.weight.data.copy_(torch.from_numpy(np_embedding))

    # if args.d_word_vec != 300:

    #     if args.embedding != None and os.path.isfile(args.embedding):
    #         np_embedding = Utils.loadFrPickle(args.embedding)
    #         embedding.weight.data.copy_(torch.from_numpy(
    #             np.array(np_embedding.weight.detach().numpy())))
    #     # else:
    #     # 	np_embedding = embedding
    #     # 	Utils.saveToPickle(args.embedding, np_embedding)
    # embedding.norm_type = 2.0
    # embedding.weight.requires_grad = False

    if args.gpu != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # os.environ[' OMP_NUM_THREADS'] = 10

    model = TCMP_pre(args=args)
    # word to vec
    # wordenc = Modules.wordEncoder(embedding=embedding)
    # # sent to vec
    # sentenc = Modules.sentEncoder(
    #     d_input=args.d_word_vec, d_output=args.d_hidden_low)
    # audioenc = Modules.sentEncoder(
    #     d_input=args.d_audio_vec, d_output=args.d_hidden_low)
    # if args.sentEnc == 'gru2' or args.sentEnc == 'unigru2':
    #     print("Utterance encoder: GRU2")
    #     sentenc = Modules.sentGRUEncoder(
    #         d_input=args.d_word_vec, d_output=args.d_hidden_low)
    #     audioenc = Modules.sentGRUEncoder(
    #         d_input=args.d_audio_vec, d_output=args.d_hidden_low)
    # if args.sentEnc == 'Trans':
    #     print("Utterance encoder: Trans Encoder")
    #     sentenc = Modules.sentTransEncoder(
    #         d_input=args.d_word_vec, d_output=args.d_hidden_low)
    #     audioenc = Modules.sentTransEncoder(
    #         d_input=args.d_audio_vec, d_output=args.d_hidden_low)

    # if args.layers == 2:
    #     print("Number of stacked GRU layers: {}".format(args.layers))
    #     sentenc = Modules.sentGRU2LEncoder(
    #         d_input=args.d_word_vec, d_output=args.d_hidden_low)

    # if args.sentEnc == 'bigru2':
    #     # cont
    #     contenc = Modules.contEncoder(
    #         d_input=args.d_hidden_low, d_output=args.d_hidden_up, bidirectional=False)
    #     contAudenc = Modules.contEncoder(
    #         d_input=args.d_hidden_low, d_output=args.d_hidden_up, bidirectional=False)
    #     # dec
    #     cmdec = Modules.biLM(d_input=args.d_hidden_up * 2,
    #                          d_output=args.d_hidden_low)
    #     cmdecAud = Modules.biLM(
    #         d_input=args.d_hidden_up * 2, d_output=args.d_hidden_low)
    # if args.sentEnc == 'unigru2':
    #     # cont
    #     contenc = Modules.unicontEncoder(
    #         d_input=args.d_hidden_low, d_output=args.d_hidden_up, bidirectional=True)
    #     contAudenc = Modules.unicontEncoder(
    #         d_input=args.d_hidden_low, d_output=args.d_hidden_up, bidirectional=True)
    #     # dec
    #     cmdec = Modules.uniLMDEC(
    #         d_input=args.d_hidden_up, d_output=args.d_hidden_low)
    #     cmdecAud = Modules.uniLMDEC(
    #         d_input=args.d_hidden_up, d_output=args.d_hidden_low)
    #     cmdecA2T = Modules.uniCrossDEC(
    #         d_input=args.d_hidden_up, d_output=args.d_hidden_low)
    #     cmdecT2A = Modules.uniCrossDEC(
    #         d_input=args.d_hidden_up, d_output=args.d_hidden_low)
    # if args.sentEnc == 'trans':
    #     # cont
    #     sentenc = Modules.sentTransEncoder(
    #         d_input=args.d_word_vec, d_output=args.d_hidden_low)
    #     audioenc = Modules.sentTransEncoder(
    #         d_input=args.d_audio_vec, d_output=args.d_hidden_low)
    #     contenc = Modules.unicontEncoder(
    #         d_input=args.d_hidden_low, d_output=args.d_hidden_up, bidirectional=True)
    #     contAudenc = Modules.unicontEncoder(
    #         d_input=args.d_hidden_low, d_output=args.d_hidden_up, bidirectional=True)
    #     cmdec = Modules.uniLMDEC(
    #         d_input=args.d_hidden_up, d_output=args.d_hidden_low)
    #     cmdecAud = Modules.uniLMDEC(
    #         d_input=args.d_hidden_up, d_output=args.d_hidden_low)
    #     cmdecA2T = Modules.uniCrossDEC(
    #         d_input=args.d_hidden_up, d_output=args.d_hidden_low)
    #     cmdecT2A = Modules.uniCrossDEC(
    #         d_input=args.d_hidden_up, d_output=args.d_hidden_low)
    # loss
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer, _ = Utils.prepare_optimizer_for_pretrained(args, model)

    # train
    # lmtrain(wordenc=wordenc, sentenc=sentenc, audioenc=audioenc, contenc=contenc, contAudenc=contAudenc, dec=cmdec,
    #         decAud=cmdecAud, cmdecA2T=cmdecA2T, cmdecT2A=cmdecT2A, criterion=criterion, data_loader=field, args=args)
    lmtrain(model=model, criterion=criterion,
            optimizer=optimizer, data_loader=field, args=args)

    # test
    print("Load best models for testing!")

    model = torch.load(args.movie_checkpoint)

    # wordenc = torch.load(args.save_dir+"/wordenc"+str(args.d_hidden_up) +
    #                      "_"+str(args.d_hidden_low)+"_"+args.dataset+".pt")
    # sentenc = torch.load(args.save_dir+"/sentenc"+str(args.d_hidden_up) +
    #                      "_"+str(args.d_hidden_low)+"_"+args.dataset+".pt")
    # contenc = torch.load(args.save_dir+"/contenc"+str(args.d_hidden_up) +
    #                      "_"+str(args.d_hidden_low)+"_"+args.dataset+".pt")
    # cmdec = torch.load(args.save_dir+"/cmdec"+str(args.d_hidden_up) +
    #                    "_"+str(args.d_hidden_low)+"_"+args.dataset+".pt")

    # audioenc = torch.load(args.save_dir+"/audioenc"+str(args.d_hidden_up) +
    #                       "_"+str(args.d_hidden_low)+"_"+args.dataset+".pt")
    # contAudenc = torch.load(args.save_dir+"/contAudenc"+str(args.d_hidden_up) +
    #                         "_"+str(args.d_hidden_low)+"_"+args.dataset+".pt")
    # cmdecAud = torch.load(args.save_dir+"/decAud"+str(args.d_hidden_up) +
    #   "_"+str(args.d_hidden_low)+"_"+args.dataset+".pt")
    with torch.no_grad():
        topkns = lmeval(model=model, data_loader=test_loader, args=args)
        # topkns = lmeval(wordenc, sentenc, audioenc, contenc, contAudenc,
        #                 cmdec, cmdecAud, cmdecA2T, cmdecT2A, test_loader, args)
    print("Validate: R1@5 R2@5 R1@11 R2@11 {}".format(topkns))

    # record the test results
    record_file = args.record_file
    if os.path.isfile(record_file):
        f_rec = open(record_file, "a")
    else:
        f_rec = open(record_file, "w")
    f_rec.write(str(datetime.now()) + "\t:\t" + str(topkns) + "\n")
    f_rec.close()


if __name__ == '__main__':
    main()
