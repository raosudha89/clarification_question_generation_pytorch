from helper_utility import *
import numpy as np
import torch
from constants import *


def evaluate(context_model, question_model, answer_model, utility_model, dev_data, criterion, args):
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0    

    context_model.eval()
    question_model.eval()
    answer_model.eval()
    utility_model.eval()
    
    with torch.no_grad():
        contexts, context_lens, questions, question_lens, answers, answer_lens, labels = dev_data
        context_masks = get_masks(context_lens, args.max_post_len)
        question_masks = get_masks(question_lens, args.max_ques_len)
        answer_masks = get_masks(answer_lens, args.max_ans_len)
        contexts = np.array(contexts)
        questions = np.array(questions)
        answers = np.array(answers)
        labels = np.array(labels)
        for c, cm, q, qm, a, am, l in iterate_minibatches(contexts, context_masks,
                                                          questions, question_masks,
                                                          answers, answer_masks,
                                                          labels, args.batch_size):
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

            # q_out: (sent_len, batch_size, num_directions*HIDDEN_DIM)
            q_hid, q_out = question_model(torch.transpose(q, 0, 1))
            qm = torch.transpose(qm, 0, 1).unsqueeze(2)
            qm = qm.expand(qm.shape[0], qm.shape[1], 2*HIDDEN_SIZE)
            q_out = torch.sum(q_out * qm, dim=0)

            a_hid, a_out = answer_model(torch.transpose(a, 0, 1))
            am = torch.transpose(am, 0, 1).unsqueeze(2)
            am = am.expand(am.shape[0], am.shape[1], 2*HIDDEN_SIZE)
            a_out = torch.sum(a_out * am, dim=0)

            predictions = utility_model(torch.cat((c_out, q_out, a_out), 1)).squeeze(1)
            predictions = torch.nn.functional.sigmoid(predictions)

            l = torch.FloatTensor([float(lab) for lab in l])
            if USE_CUDA:
                l = l.cuda()
            loss = criterion(predictions, l)
            acc = binary_accuracy(predictions, l)
            epoch_loss += loss.item()
            epoch_acc += acc
            num_batches += 1
        
    return epoch_loss / num_batches, epoch_acc / num_batches

