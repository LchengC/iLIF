import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import pickle
from process_event import process_node

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

def setup_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--y_class", default=3, type=int)
    parser.add_argument('--seed', default=209, type=int)
    parser.add_argument('--len_arg', default=105, type=int)
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    torch.cuda.empty_cache()
    # set seed for random number
    setup_seed(args.seed)

    # Enter your existing data here
    data_document = np.load('./XXX.npy', allow_pickle=True).item()
    features_train = []
    features_dev = []

    for topic in data_document:
        for doc in data_document[topic]:
            enc_text = []
            document = data_document[topic][doc][0]
            topic_id = topic
            doc_id = doc
            for s in document[0]:
                enc_text += s
            enc_text = " ".join(enc_text)
            sentences = document[0]
            enc_tokens, enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, event_pairs = process_node(document, args)

            doc_features = ESC_features(topic_id, doc_id, enc_text, enc_tokens, sentences,
                                        enc_input_ids, enc_mask_ids, node_event,
                                        t1_pos, t2_pos, target, rel_type, event_pairs
                                        )
            if topic != '37' and topic != '41':
                features_train.append(doc_features)
            else:
                features_dev.append(doc_features)

    with open('./database/EventStoryLine_bert_intra_and_inter_train_features.pkl', 'wb') as f_t:
        pickle.dump(features_train, f_t)
    with open('./database/EventStoryLine_bert_intra_and_inter_dev_features.pkl', 'wb') as f_d:
        pickle.dump(features_dev, f_d)

if __name__ == "__main__":
    main()