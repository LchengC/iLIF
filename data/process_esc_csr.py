import os
import pickle
from torch.utils.data import Dataset

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

class ESC_dataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def collate_fn(self, batch):
        enc_input_ids = batch[0].enc_input_ids
        enc_mask_ids = batch[0].enc_mask_ids
        node_event = batch[0].node_event
        t1_pos = [f.t1_pos for f in batch]
        t2_pos = [f.t2_pos for f in batch]
        target = [f.target for f in batch]
        rel_type = [f.rel_type for f in batch]
        event_pairs = batch[0].event_pairs
        return (enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, event_pairs)

class ESC_processor(object):
    def __init__(self, args, tokenizer, printlog):
        self.args = args
        self.tokenizer = tokenizer
        self.printlog = printlog
    def convert_features_to_dataset(self, features):
        dataset = ESC_dataset(features)
        return dataset

    def load_and_cache_features(self, cache_path):
        features = pickle.load(open(cache_path, 'rb'))
        self.printlog(f"load features from {cache_path}")
        return features

    def generate_dataloader(self, set_type):
        assert (set_type in ['train', 'dev'])
        cache_feature_path = os.path.join(self.args.cache_path,
                                          "{}_{}_{}_{}_features.pkl".format(self.args.dataset_type, self.args.model_type, self.args.inter_or_intra,
                                                                         set_type))

        features = self.load_and_cache_features(cache_feature_path)

        dataset = self.convert_features_to_dataset(features)

        return features, dataset