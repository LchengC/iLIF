# coding: UTF-8

import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from .CGE import CGEConv
from torch.nn.utils.rnn import pad_sequence
embedding_size = 768

class focal_loss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += (1 - alpha)
            self.alpha[1:] += alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=-1)
        preds_logsoft = F.log_softmax(preds, dim=-1)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class bertCSRModel(nn.Module):
    def __init__(self, args):
        super(bertCSRModel, self).__init__()
        self.device = args.device
        self.pretrained_model = BertModel.from_pretrained(args.model_name)
        self.mlp = nn.Sequential(nn.Linear(1 * args.n_last + 2 * embedding_size, args.mlp_size),
                                  nn.ReLU(), nn.Dropout(args.mlp_drop),
                                  nn.Linear(args.mlp_size, 3))
        self.rate = args.rate
        self.w = args.w
        self.max_iteration = args.max_iteration + 1
        self.threshold = args.threshold
        self.min_iteration = args.min_iteration
        self.CGE = CGEConv(in_channels=args.in_channels,
                            out_channels=args.n_last,
                            metadata=args.metadata,
                            heads=args.num_heads,
                            beta_intra=args.beta_intra,
                            beta_inter=args.beta_inter
                            )
        self.focal_loss = focal_loss(gamma=args.gamma, num_classes=args.no_of_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_type = args.loss_type

    def forward(self, enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, event_pairs):
        sent_emb = self.pretrained_model(enc_input_ids, enc_mask_ids)[0].to(self.device)
        sent_list = []
        for idx, event in enumerate(node_event):
            sent_id = int(event[0])
            event_pid = event[1]
            e_emb = self.extract_event(sent_emb[sent_id], event_pid).to(self.device)
            if idx == 0:
                event_embed = e_emb
            else:
                event_embed = torch.cat((event_embed, e_emb))
            if sent_id not in sent_list:
                sent_list.append(sent_id)
        if self.max_iteration > sent_list.__len__():
            max_iteration = sent_list.__len__()
        else:
            max_iteration = self.max_iteration

        pad_event1_pos = pad_sequence([torch.tensor(pos) for pos in t1_pos]).t().to(self.device).long()
        pad_event2_pos = pad_sequence([torch.tensor(pos) for pos in t2_pos]).t().to(self.device).long()
        event1 = torch.index_select(event_embed, 0, pad_event1_pos[0])
        event2 = torch.index_select(event_embed, 0, pad_event2_pos[0])
        event_pair_embed = torch.cat([event1, event2], dim=-1)

        loss = 0.0
        iteration = 0
        difference = self.threshold
        event_embed = self.CGE.proj['event'](event_embed)
        if len(event_pairs) > 1:
            target = torch.cat([torch.tensor(t) for t in target], dim=0).to(self.device)
        else:
            target = torch.tensor(target[0]).to(self.device)
        while iteration < max_iteration and (iteration < self.min_iteration or difference > self.threshold):
            if len(event_pairs) > 1:
                event_diff = torch.cat([event_embed[[pair[0]]] - event_embed[[pair[1]]] for pair in event_pairs])
            else:
                event_diff = event_embed[[event_pairs[0][0]]] - event_embed[[event_pairs[0][1]]]
            event_pair_pre = torch.cat((event_pair_embed, event_diff), dim=1)
            prediction = self.mlp(event_pair_pre)
            if self.loss_type == 'focal':
                loss += (1/(iteration + 1)) * self.focal_loss(prediction, target)
            else:
                loss += (1/(iteration + 1)) * self.ce_loss(prediction, target)
            graphedge_index = self.get_graphedge_index(prediction, event_pairs, rel_type)
            if graphedge_index.__len__() > 0:
                node_length = event_embed.size()[0]
                self_loop = torch.cat((torch.arange(node_length).unsqueeze(0), torch.arange(node_length).unsqueeze(0)),dim=0).to(self.device)
                if ('event', 'intra', 'event') in graphedge_index:
                    graphedge_index[('event', 'intra', 'event')] = torch.cat((graphedge_index[('event', 'intra', 'event')], self_loop), dim=-1).long()
                else:
                    graphedge_index[('event', 'intra', 'event')] = self_loop.long()
                event_han = {'event': event_embed}
                event_han = self.CGE(event_han, graphedge_index)
                event_embed = event_han['event']
            else:
                event_embed = self.CGE.proj['event'](event_embed)
            if iteration > 0:
                difference = self.Contrast_pre(prediction, prediction_1)
            iteration += 1
            prediction_1 = prediction
        return loss, prediction

    # Extract the representation vector of the event
    def extract_event(self, embed, event_pid):
        e_1 = int(event_pid[0])
        e_2 = int(event_pid[1])
        e1_embed = torch.zeros(1, embedding_size).to(self.device)
        length = e_2 - e_1
        for j in range(e_1,e_2):
            e1_embed += embed[j]
        event_embed = e1_embed / length
        return event_embed

    # Construct adjacency matrix
    def get_graphedge_index(self, prediction, event_pair, rel_type):
        graphedge_index = {}
        if self.training:
            rate = self.rate
        else:
            rate = self.w
        if rate != 0:
            pred_soft = torch.softmax(prediction, dim=1)
            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and rel_type[0][j] == 0)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and rel_type[0][j] == 0)]).t().to(self.device)
            if tmp.size()[0] > 0:
                intra_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                intra_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_2 = tmp_1
            intra_graphedge_index = torch.cat((intra_graphedge_index_1, intra_graphedge_index_2), dim=-1)

            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and rel_type[0][j] == 1)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and rel_type[0][j] == 1)]).t().to(self.device)
            if tmp.size()[0] > 0:
                inter_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                inter_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_2 = tmp_1
            inter_graphedge_index = torch.cat((inter_graphedge_index_1, inter_graphedge_index_2), dim=-1)
        else:
            predt = torch.argmax(prediction, dim=1)
            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (predt[j] == 1 and rel_type[0][j] == 0)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (predt[j] == 2 and rel_type[0][j] == 0)]).t().to(self.device)
            if tmp.size()[0] > 0:
                intra_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (predt[j] == 1 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                intra_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (predt[j] == 2 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_2 = tmp_1
            intra_graphedge_index = torch.cat((intra_graphedge_index_1, intra_graphedge_index_2), dim=-1)

            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (predt[j] == 1 and rel_type[0][j] == 1)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (predt[j] == 2 and rel_type[0][j] == 1)]).t().to(self.device)
            if tmp.size()[0] > 0:
                inter_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (predt[j] == 1 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                inter_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (predt[j] == 2 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_2 = tmp_1
            inter_graphedge_index = torch.cat((inter_graphedge_index_1, inter_graphedge_index_2), dim=-1)
        if intra_graphedge_index.size()[0] > 0:
            graphedge_index[('event', 'intra', 'event')] = intra_graphedge_index.long()
        if inter_graphedge_index.size()[0] > 0:
            graphedge_index[('event', 'inter', 'event')] = inter_graphedge_index.long()
        return graphedge_index

    # Compare the structural differences before and after iteration
    def Contrast_pre(self, prediction, prediction_last):
        if self.training:
            rate = self.rate
        else:
            rate = self.w
        if rate != 0:
            pred_soft = torch.softmax(prediction, dim=1)
            pred_last_soft = torch.softmax(prediction_last, dim=1)
            pre_list = torch.tensor([]).to(self.device)
            pre_last_list = torch.tensor([]).to(self.device)
            max_pro, pred_t = torch.max(pred_soft, dim=1)
            for idx, pre in enumerate(max_pro):
                if pre > self.rate:
                    pre_list = torch.cat((pre_list, pred_t[[idx]]), dim=-1)
                else:
                    pre_list = torch.cat((pre_list, torch.tensor([0]).to(self.device)), dim=-1)
            max_pro, pred_t = torch.max(pred_last_soft, dim=1)
            for idx, pre in enumerate(max_pro):
                if pre > self.rate:
                    pre_last_list = torch.cat((pre_last_list, pred_t[[idx]]), dim=-1)
                else:
                    pre_last_list = torch.cat((pre_last_list, torch.tensor([0]).to(self.device)), dim=-1)
        else:
            pre_list = torch.argmax(prediction, dim=1)
            pre_last_list = torch.argmax(prediction_last, dim=1)
        different = (pre_list != pre_last_list).sum().item()
        return different