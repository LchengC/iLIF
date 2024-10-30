import torch
from transformers import BertTokenizer

def delete_tokens(arg_idx, arg_mask, index, idx):
    arg_idx = torch.LongTensor(arg_idx)
    temp = torch.nonzero(arg_idx == idx, as_tuple=False)
    indices = temp[index][1]
    arg_i = torch.cat((arg_idx[0][0:indices], arg_idx[0][indices + 1:]))
    arg_i = torch.unsqueeze(arg_i, dim=0)
    arg_m = torch.cat((arg_mask[0][0:indices], arg_mask[0][indices + 1:]))
    arg_m = torch.unsqueeze(arg_m, dim=0)
    return arg_i, arg_m, indices

def process_node(arg, args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    doc = arg[0]
    event_nodes = arg[1]
    doc_pair = arg[2]
    enc_tokens = []
    enc_input_ids = []
    enc_mask_ids = []
    node_event = []
    event_pairs = []
    target = []
    rel_type = []
    t1_pos = []
    t2_pos = []
    for sent in doc:
        enc_tokens += sent
    enc_tokens = " ".join(enc_tokens)
    encode_dict = tokenizer.encode_plus(
        enc_tokens,
        add_special_tokens=False)
    enc_tokens = encode_dict['input_ids']

    #####Encode document content#####
    for idx, sent in enumerate(doc):
        s = " ".join(sent)
        encode_dict = tokenizer.encode_plus(
            s,
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']
        if idx == 0:
            enc_input_ids = arg_1_idx
            enc_mask_ids = arg_1_mask
        else:
            enc_input_ids = torch.cat((enc_input_ids, arg_1_idx), dim=0)
            enc_mask_ids = torch.cat((enc_mask_ids, arg_1_mask), dim=0)

    #####Get the event information list，event_pos = [[sent_id, event_pos_id], ...]#####
    for idx, event_node in enumerate(event_nodes):
        assert idx == int(event_node[2])
        e_id = event_node[5]
        e_id = e_id.split("_")
        s_id = int(event_node[4]) - 1
        assert s_id >= 0
        s = doc[s_id].copy()
        s.insert(int(e_id[1]), '[CLS]')
        s.insert(int(e_id[1]) + len(e_id), '[CLS]')
        s = " ".join(s)
        encode_dict = tokenizer.encode_plus(
            s,
            add_special_tokens=True,
            padding='max_length',
            max_length=120,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']
        arg_1_idx, arg_1_mask, v1 = delete_tokens(arg_1_idx, arg_1_mask, 1, 101)
        arg_1_idx, arg_1_mask, v2 = delete_tokens(arg_1_idx, arg_1_mask, 1, 101)
        node_event.append([s_id, [v1,v2]])

    #####Gets a list of event pairs:[e1_id, e2_id]、label、clabel######
    for idx, pair in enumerate(doc_pair):
        sentenceOf1 = int(pair[7])
        sentenceOf2 = int(pair[8])
        relation = pair[6]
        if relation == 'NONE':
            target.append(0)
        elif relation == 'PRECONDITION':
            target.append(1)
        else:
            target.append(2)
        if sentenceOf1 != sentenceOf2:
            rel_type.append(1)
        else:
            rel_type.append(0)
        e_1 = pair[2]
        e_2 = pair[3]
        event_pairs.append([e_1,e_2])
        t1_pos.append(e_1)
        t2_pos.append(e_2)
    return enc_tokens, enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, event_pairs