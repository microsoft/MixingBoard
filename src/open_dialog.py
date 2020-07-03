#// Copyright (c) Microsoft Corporation.// Licensed under the MIT license. 

import torch, os, pdb
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, BertTokenizer
import time
from todo import pick_tokens


class DialoGPT:

    def __init__(self, use_cuda=True, path_model='models/DialoGPT/medium_ft.pkl'):
        self.use_cuda = use_cuda
        self.turn_sep = ' <|endoftext|> '
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model_config = GPT2Config(n_embd=1024, n_layer=24, n_head=16)        
        self.model = GPT2LMHeadModel(model_config)
        weights = torch.load(path_model)
        weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
        weights.pop("lm_head.decoder.weight",None)
        self.model.load_state_dict(weights)
        if self.use_cuda:
            self.model = self.model.cuda()
        self.ix_EOS = 50256
        self.way = 'DPT'
        self.model.eval()


    def tf_prob(self, context, hyps, use_EOS=True, verbose=False, batch=10, return_np=True):
        if isinstance(hyps, str):
            hyps = [hyps]
        i0 = 0
        prob = []
        while i0 < len(hyps):
            i1 = min(i0 + batch, len(hyps))
            with torch.no_grad():
                prob.append(self._tf_prob(context, hyps[i0:i1], use_EOS=use_EOS))
            i0 = i1
        if len(prob) > 1:
            prob = torch.cat(prob, dim=0)
        else:
            prob = prob[0]
        if return_np:
            if self.use_cuda:
                prob = prob.cpu()
            return prob.detach().numpy()
        else:
            return prob
            

    def _tf_prob(self, context, hyps, use_EOS=True):
        # converted what's from tokenizer.encode to what's should be used in logits
        enc2pred = {
            11:837,     # ',' => 'Ġ,'
            13:764,     # '.' => 'Ġ.'
            0:5145,     # '!' => 'Ġ!'
            30:5633,     # '?' => 'Ġ?'
            }
        ids_cxt = self.tokenizer.encode(context) + [self.ix_EOS]
        ids_hyp = []
        hyp_len = []
        for hyp in hyps:
            hyp = (' ' + hyp + ' ').replace(' i ',' I ')
            hyp = hyp.strip().replace(" '","'")
            hyp = hyp[0].upper() + hyp[1:]
            raw_hyp_tokens = self.tokenizer.encode(hyp)
            
            # if not use_EOS, then hyps are some incomplete hyps, as in decoding with cross-model scoring
            if use_EOS:
                raw_hyp_tokens.append(self.ix_EOS)
            hyp_tokens = []
            for token in raw_hyp_tokens:
                hyp_tokens.append(enc2pred.get(token, token))
            ids_hyp.append(hyp_tokens)
            hyp_len.append(len(hyp_tokens))
        
        max_len = max(hyp_len)
        ids = []
        mask = []
        for i, seq in enumerate(ids_hyp):
            cat = ids_cxt + seq + [self.ix_EOS] * (max_len - hyp_len[i])
            ids.append(cat)
            mask.append([1] * hyp_len[i] + [0] * (max_len - hyp_len[i]))
        ids = torch.tensor(ids)
        mask = torch.FloatTensor(mask)
        hyp_len = torch.FloatTensor(hyp_len)
        if self.use_cuda:
            ids = ids.to('cuda')
            mask = mask.to('cuda')
            hyp_len = hyp_len.to('cuda')
        
        l_cxt = len(ids_cxt)
        with torch.no_grad():
            logits, _ = self.model(ids)
            logits = logits[:, l_cxt - 1: -1, :]     # only care the part after cxt. ignore -1.
            logP = torch.log(F.softmax(logits, dim=-1))

        logP_ids = logP.gather(dim=-1, index=ids[:,l_cxt:].unsqueeze(-1)).squeeze(-1)
        avg_logP = (logP_ids * mask).sum(dim=-1) / hyp_len
        return torch.exp(avg_logP)
    
    
    def rvs_prob(self, cxt, hyps, batch=10):
        i0 = 0
        prob = []
        while i0 < len(hyps):
            i1 = min(i0 + batch, len(hyps))
            with torch.no_grad():
                prob.append(self._rvs_prob(cxt, hyps[i0:i1]))
            i0 = i1
        return np.concatenate(prob, axis=0)
            

    def _rvs_prob(self, context, hyps):
        # converted what's from tokenizer.encode to what's should be used in logits
        enc2pred = {
            11:837,     # ',' => 'Ġ,'
            13:764,     # '.' => 'Ġ.'
            0:5145,     # '!' => 'Ġ!'
            30:5633,     # '?' => 'Ġ?'
            }
        
        raw_ids_cxt = self.tokenizer.encode(context) + [self.ix_EOS]
        ids_cxt = []
        for token in raw_ids_cxt:
            ids_cxt.append(enc2pred.get(token, token))

        ids_hyp = []
        hyp_len = []
        for hyp in hyps:
            hyp = (' ' + hyp + ' ').replace(' i ',' I ')
            hyp = hyp.strip().replace(" '","'")
            hyp = hyp[0].upper() + hyp[1:]

            hyp_tokens = self.tokenizer.encode(hyp) + [self.ix_EOS]
            ids_hyp.append(hyp_tokens)
            hyp_len.append(len(hyp_tokens))
        
        max_len = max(hyp_len)
        ids = []
        for i, seq in enumerate(ids_hyp):
            cat = seq + ids_cxt + [self.ix_EOS] * (max_len - hyp_len[i])
            ids.append(cat)
        ids = torch.tensor(ids)
        if self.use_cuda:
            ids = ids.to('cuda')
        with torch.no_grad():
            logits, _ = self.model(ids)
            logP = torch.log(F.softmax(logits, dim=-1))

        logP_cxt = []
        for i, l in enumerate(hyp_len):
            _logP = []
            for t, token in enumerate(ids_cxt):
                _logP.append(logP[i, l + t - 1, token].item())
            logP_cxt.append(np.mean(_logP))
        return np.exp(logP_cxt)


    def predict(self, context, beam=10, branch=2, verbose=False):
        # return n hypotheses given context, in parallel
        # context is str
        way = self.way

        conditioned_tokens = self.tokenizer.encode(context) + [self.ix_EOS]
        len_cxt = len(conditioned_tokens)
        tokens = torch.tensor([conditioned_tokens]).view(1, -1)
        if self.use_cuda:
            tokens = tokens.cuda()

        finished = []
        hyp_set = set()
        sum_logP = [0]
        max_t = 30
        for t in range(max_t):
            with torch.no_grad():
                outputs = self.model(tokens)
                predictions = outputs[0]
            logits = predictions[:, -1, :]              # only care the last step. [n_hyp, vocab]
            prob = F.softmax(logits, dim=-1)
            logP = torch.log(prob)
            picked_tokens = pick_tokens(prob, branch)

            cand = []
            #tokens_np = (tokens.cpu() if self.use_cuda else tokens).detach().numpy()
            for i in range(picked_tokens.shape[0]):
                for j in range(picked_tokens.shape[1]):
                    ix = picked_tokens[i, j].item()
                    _sum_logP = sum_logP[i] + logP[i, ix].item()
                    cand.append((_sum_logP, i, j))

            cand = sorted(cand, reverse=True)
            cand = cand[:min(len(cand), beam)]
                
            sum_logP = []
            cur = []
            nxt = []
            for _sum_logP, i, j in cand:
                ix = picked_tokens[i, j].item()
                if ix == self.ix_EOS:
                    seq = [w.item() for w in tokens[i, len_cxt: len_cxt + t]]
                    seq_tup = tuple(seq)
                    if seq_tup not in hyp_set:
                        finished.append((np.exp(_sum_logP/len(seq)), seq))
                        hyp_set.add(seq_tup)
                        continue

                cur.append(tokens[i:i+1,:])
                nxt.append(picked_tokens[i:i+1, j])
                sum_logP.append(_sum_logP)
                if len(cur) == beam:
                    break
            
            tokens = torch.cat([torch.cat(cur, dim=0), torch.cat(nxt, dim=0).unsqueeze(-1)], dim=-1)

        finished = sorted(finished, reverse=True)
        ret = []
        for _sum_logP, seq in finished:
            prob = np.exp(_sum_logP/len(seq))
            hyp = self.tokenizer.decode(seq).strip()
            ret.append((way, prob, hyp))
            if len(ret) == branch:
                break
        return sorted(ret, reverse=True)



if __name__ == "__main__":
    dialogpt = DialoGPT()
    while True:
        cxt = input('\nCONTEXT:\t')
        ret = dialogpt.predict(cxt)
        for way, prob, hyp in ret:
            print('%s %.3f\t%s'%(way, prob, hyp))