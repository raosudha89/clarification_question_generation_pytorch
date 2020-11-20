from constants import *
import sys
sys.path.append('src/utility')
from helper_utility import *
import numpy as np
import torch
from torch.autograd import Variable


def evaluate_utility(context_model, question_model, answer_model, utility_model, c, cl, q, ql, a, al, args):
    with torch.no_grad():
        context_model.eval()
        question_model.eval()
        answer_model.eval()
        utility_model.eval()
        cm = get_masks(cl, args.max_post_len)
        qm = get_masks(ql, args.max_ques_len)
        am = get_masks(al, args.max_ans_len)

        c = torch.tensor(c)
        cm = torch.FloatTensor(cm)
        q = torch.tensor(q)
        qm = torch.FloatTensor(qm)
        a = torch.tensor(a)
        am = torch.FloatTensor(am)
        if USE_CUDA:
            c = c.cuda()
            cm = cm.cuda()
            q = q.cuda()
            qm = qm.cuda()
            a = a.cuda()
            am = am.cuda()
        c_hid, c_out = context_model(torch.transpose(c, 0, 1))
        cm = torch.transpose(cm, 0, 1).unsqueeze(2)
        cm = cm.expand(cm.shape[0], cm.shape[1], 2*HIDDEN_SIZE)
        c_out = torch.sum(c_out * cm, dim=0)

        q_hid, q_out = question_model(torch.transpose(q, 0, 1))
        qm = torch.transpose(qm, 0, 1).unsqueeze(2)
        qm = qm.expand(qm.shape[0], qm.shape[1], 2*HIDDEN_SIZE)
        q_out = torch.sum(q_out * qm, dim=0)

        a_hid, a_out = answer_model(torch.transpose(a, 0, 1))
        am = torch.transpose(am, 0, 1).unsqueeze(2)
        am = am.expand(am.shape[0], am.shape[1], 2*HIDDEN_SIZE)
        a_out = torch.sum(a_out * am, dim=0)

        predictions = utility_model(torch.cat((c_out, q_out, a_out), 1)).squeeze(1)
        # predictions = utility_model(torch.cat((c_out, q_out), 1)).squeeze(1)
        # predictions = utility_model(q_out).squeeze(1)
        predictions = torch.nn.functional.sigmoid(predictions)

    return predictions
