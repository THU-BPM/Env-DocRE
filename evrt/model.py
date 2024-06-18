import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ClsLoss


class DocREModel(nn.Module):
    def __init__(self, args, config, model, block_size=64):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fnt = ClsLoss(shift=args.clsloss_shift, regularization=args.clsloss_reg)

        self.head_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.bilinear = nn.Linear(config.hidden_size * block_size, config.num_labels)

        self.emb_size = config.hidden_size
        self.block_size = block_size
        self.num_labels = args.num_labels

    def encode(self, input_ids, attention_mask):
        config = self.config
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta", "deberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])  # [n_e*(n_e-1), d]
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])  # [n_e*(n_e-1), d]

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])  # [n_e*(n_e-1), h, seq_len]
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])  # [n_e*(n_e-1), h, seq_len]
            ht_att = (h_att * t_att).mean(1)  # [n_e*(n_e-1), seq_len]
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)  # [n_e*(n_e-1), seq_len]
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)  # [n_e*(n_e-1), d]
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)  # [b*n_e*(n_e-1), d]
        tss = torch.cat(tss, dim=0)  # [b*n_e*(n_e-1), d]
        rss = torch.cat(rss, dim=0)  # [b*n_e*(n_e-1), d]
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))  # [b*n_e*(n_e-1), d]
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))  # [b*n_e*(n_e-1), d]
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)  # [b*n_e*(n_e-1), num_classes]

        output = (
            self.loss_fnt.get_label(logits, num_labels=self.num_labels), logits, bl,)  # [b*n_e*(n_e-1), num_classes]
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output
