# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, RandomSampler, SequentialSampler, DataLoader
import logging
import tqdm
from datetime import datetime
from data import ESC_processor
from utils import transfor3to2, compute_f1, setup_seed, calculate
from pathlib import Path
from sklearn.model_selection import KFold
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertTokenizer
from parameter import parse_args

from model import bertCSRModel

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, bertCSRModel, BertTokenizer)
}

class ESC_features(object):
    def __init__(self, topic_id, doc_id,
                 enc_text, enc_tokens, sentences,
                 enc_input_ids, enc_mask_ids, node_event,
                 t1_pos, t2_pos, target, rel_type, event_pairs
                 ):
        self.topic_id = topic_id
        self.doc_id = doc_id
        self.enc_text = enc_text
        self.enc_tokens = enc_tokens
        self.sentences = sentences
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.node_event = node_event
        self.t1_pos = t1_pos
        self.t2_pos = t2_pos
        self.target = target
        self.rel_type = rel_type
        self.event_pairs = event_pairs

def train_epoch(args, model, train_loader, optimizer, scheduler, epoch):
    train_loss = []
    predicted_all = []
    gold_all = []
    clabel_all = []
    model.train()
    progress = tqdm.tqdm(total=len(train_loader), ncols=75,
                         desc='Train {}'.format(epoch))
    for step, batch in enumerate(train_loader):
        progress.update(1)
        inputs = {'enc_input_ids': batch[0].to(args.device),
                  'enc_mask_ids': batch[1].to(args.device),
                  'node_event': batch[2],
                  't1_pos': batch[3],
                  't2_pos': batch[4],
                  'target': batch[5],
                  'rel_type': batch[6],
                  'event_pairs': batch[7]
                  }
        loss, opt = model(**inputs)
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        if step % args.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        train_loss.append(loss.item())
        predicted_ = torch.argmax(opt, -1)
        predicted_ = list(predicted_.cpu().numpy())
        gold_ = [t for bt in inputs['target'] for t in bt]
        clabel = [t for bt in inputs['rel_type'] for t in bt]
        # transform the three classification results predicted by the model to Causality Existence Identification and Causality Direction Identification results
        # predicted: the Identification Result of Causality Direction(we calculate the micro-averaged results for Precision, Recall, and F1-score specifically for the CAUSE and EFFECT classes)
        gold, predicted = transfor3to2(gold_, predicted_)
        gold_all += gold
        predicted_all += predicted
        clabel_all += clabel
        p, r, f1 = compute_f1(gold_all, predicted_all, logger)
        if step % args.logging_steps == 0:
            printlog('Step {}: Train Loss {} P {:.4f} R {:.4f} F1 {:.4f}'.format(step, np.mean(train_loss), p, r, f1))
    progress.close()
    p, r, f1 = compute_f1(gold_all, predicted_all, logger)
    _, _, _, intra, cross = calculate(gold_all, predicted_all, clabel_all, epoch, printlog)

    return np.mean(train_loss), r, p, f1, intra, cross

def test_epoch(args, model, test_loader, epoch):
    test_loss = []
    predicted_all = []
    gold_all = []
    clabel_all = []
    preds = []
    golds = []
    clabels = []
    for batch in test_loader:
        inputs = {'enc_input_ids': batch[0].to(args.device),
                  'enc_mask_ids': batch[1].to(args.device),
                  'node_event': batch[2],
                  't1_pos': batch[3],
                  't2_pos': batch[4],
                  'target': batch[5],
                  'rel_type': batch[6],
                  'event_pairs': batch[7]
                  }
        loss, opt = model(**inputs)
        test_loss.append(loss.item())
        predicted_ = torch.argmax(opt, -1)
        predicted_ = list(predicted_.cpu().numpy())
        gold_ = [t for bt in inputs['target'] for t in bt]
        clabel = [t for bt in inputs['rel_type'] for t in bt]
        # transform the three classification results predicted by the model to Causality Existence Identification and Causality Direction Identification results
        # predicted: the Identification Result of Causality Direction(we calculate the micro-averaged results for Precision, Recall, and F1-score specifically for the CAUSE and EFFECT classes)
        gold, predicted = transfor3to2(gold_, predicted_)
        predicted_all += predicted
        preds.append(predicted)
        gold_all += gold
        clabel_all += clabel
        golds.append(gold)
        clabels.append(clabel)

    p, r, f1 = compute_f1(gold_all, predicted_all, logger)
    _, _, _, intra, cross = calculate(gold_all, predicted_all, clabel_all, epoch, printlog)

    return preds, golds, clabels, np.mean(test_loss), r, p, f1, intra, cross

def valid_epoch(args, model, valid_loader, epoch):
    valid_loss = []
    predicted_all = []
    gold_all = []
    clabel_all = []
    for batch in valid_loader:
        inputs = {'enc_input_ids': batch[0].to(args.device),
                  'enc_mask_ids': batch[1].to(args.device),
                  'node_event': batch[2],
                  't1_pos': batch[3],
                  't2_pos': batch[4],
                  'target': batch[5],
                  'rel_type': batch[6],
                  'event_pairs': batch[7]
                  }
        loss, opt = model(**inputs)
        valid_loss.append(loss.item())
        predicted_ = torch.argmax(opt, -1)
        predicted_ = list(predicted_.cpu().numpy())
        gold_ = [t for bt in inputs['target'] for t in bt]
        clabel = [t for bt in inputs['rel_type'] for t in bt]
        # transform the three classification results predicted by the model to Causality Existence Identification and Causality Direction Identification results
        # predicted: the Identification Result of Causality Direction(we calculate the micro-averaged results for Precision, Recall, and F1-score specifically for the CAUSE and EFFECT classes)
        gold, predicted = transfor3to2(gold_, predicted_)
        predicted_all += predicted
        gold_all += gold
        clabel_all += clabel
    p, r, f1 = compute_f1(gold_all, predicted_all, logger)
    _, _, _, intra, cross = calculate(gold_all, predicted_all, clabel_all, epoch, printlog)
    return np.mean(valid_loss), r, p, f1, intra, cross

def cross_validation(args, model_class, processor):
    # Divide the training data into 5 folds randomly.
    splits = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    printlog("train dataloader generation")
    train_test_features, train_test_dataset = processor.generate_dataloader('train')
    # dev data
    printlog("dev dataloader generation")
    dev_features, dev_dataset = processor.generate_dataloader('dev')
    valid_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

    avg_test_f1, avg_test_r, avg_test_p, avg_test_intra_p, avg_test_intra_r, avg_test_intra_f1, avg_test_cross_p, avg_test_cross_r, avg_test_cross_f1 = [], [], [], [], [], [], [], [], []
    # 5 folds
    for fold, (train_idx, test_idx) in enumerate(splits.split(train_test_dataset)):
        printlog("Fold {}".format(fold + 1))
        train_dataset = Subset(train_test_dataset, train_idx)
        test_dataset = Subset(train_test_dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset),
                                  collate_fn=train_test_dataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, sampler=SequentialSampler(test_dataset),
                                 collate_fn=train_test_dataset.collate_fn)

        model = model_class(args).to(args.device)
        model.to(args.device)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.t_lr},
            {'params': [p for n, p in model.pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.t_lr},

            {'params': [p for n, p in model.mlp.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.mlp.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},

            {'params': [p for n, p in model.CGE.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.CGE.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        total_steps = len(train_loader) * args.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * args.warm_up_ratio,
                                                    num_training_steps=total_steps)
        best_valid_f1 = 0
        best_valid_test_preds, best_valid_test_golds, best_valid_test_f1, best_valid_test_r, best_valid_test_p = 0, 0, 0, 0, 0
        for epoch in range(1, args.epoch + 1):
            model.zero_grad()
            train_loss, train_r, train_p, train_f1, train_intra, train_cross = train_epoch(args, model, train_loader, optimizer, scheduler, epoch)
            printlog('Epoch{}: Train Loss {} R {} P {} F {}'.format(epoch, train_loss, train_r, train_p, train_f1))
            model.eval()
            with torch.no_grad():
                if epoch % args.test_epoch == 0:
                    test_preds, test_golds, test_clabels, test_loss, test_r, test_p, test_f1, test_intra, test_cross = test_epoch(args, model, test_loader, epoch)
                    printlog('Epoch{}: Test Loss {} R {} P {} F {} intra{} cross{}'.format(epoch, test_loss, test_r, test_p, test_f1, test_intra, test_cross))
                if epoch % args.valid_epoch == 0:
                    valid_loss, valid_r, valid_p, valid_f1, valid_intra, valid_cross = valid_epoch(args, model, valid_loader, epoch)
                    printlog(
                        'Epoch{}: Valid Loss {} R {} P {} F {} intra{} cross{}'.format(epoch, valid_loss, valid_r, valid_p, valid_f1, valid_intra, valid_cross))
                    if valid_f1 >= best_valid_f1:
                        best_valid_f1 = valid_f1
                        best_valid_test_f1 = test_f1
                        best_valid_test_r = test_r
                        best_valid_test_p = test_p
                        best_valid_test_intra = test_intra
                        best_valid_test_cross = test_cross

        printlog('Fold {}: R {} P {} F {} intra{} cross{}'.format(fold + 1, best_valid_test_r, best_valid_test_p, best_valid_test_f1, best_valid_test_intra, best_valid_test_cross))
        avg_test_f1.append(best_valid_test_f1)
        avg_test_r.append(best_valid_test_r)
        avg_test_p.append(best_valid_test_p)
        avg_test_intra_p.append(best_valid_test_intra['p'])
        avg_test_intra_r.append(best_valid_test_intra['r'])
        avg_test_intra_f1.append(best_valid_test_intra['f1'])
        avg_test_cross_p.append(best_valid_test_cross['p'])
        avg_test_cross_r.append(best_valid_test_cross['r'])
        avg_test_cross_f1.append(best_valid_test_cross['f1'])


    printlog("\n-------------------------Causality Direction Identification-------------------------\n")
    printlog("--------  \t Intra: P {} R {} F1 {} \n\t Cross: P {} R {} F1 {}  \n\t 5-Fold Avg: P {} R {} F1 {}---------".format(np.mean(avg_test_intra_p), np.mean(avg_test_intra_r), np.mean(avg_test_intra_f1), np.mean(avg_test_cross_p), np.mean(avg_test_cross_r),
                                                                                                                  np.mean(avg_test_cross_f1), np.mean(avg_test_p), np.mean(avg_test_r), np.mean(avg_test_f1)))

    # Tips: Another average F1: The final F1 value can be calculated according to F1=2*P*R/(P+R) by the final P value and final R value

def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)

def main():
    args = parse_args()
    torch.cuda.empty_cache()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    args.log = Path(f"./logout/seed:{args.seed}")
    args.log.mkdir(exist_ok=True, parents=True)

    args.log = str(args.log) + '/t_lr' + str(args.t_lr) + 'lr' + str(args.lr) + t + '.txt'

    for name in logging.root.manager.loggerDict:
        if 'transformers' in name:
            logging.getLogger(name).setLevel(logging.CRITICAL)

    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=args.log,
                        filemode='w')

    setup_seed(args.seed)

    config_class, eci_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name, add_special_tokens=True)

    processor = ESC_processor(args, tokenizer, printlog)

    printlog("Training/evaluation parameters :{}".format(args))
    cross_validation(args, eci_model_class, processor)

if __name__ == "__main__":
    main()