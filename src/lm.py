from todo import pick_tokens
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch, pdb
import torch.nn.functional as F
import numpy as np


class LanguageModel:
    # based on: https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
    
    def __init__(self, use_cuda=True):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        if use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.use_cuda = use_cuda
        self.ix_EOS = 50256


    def predict(self, inp, beam=10, max_t=30):
        # num_samples=1, temperature=0., repetition_penalty=1.,top_k=0, top_p=0.9
        ids = self.tokenizer.encode(inp) 
        len_cxt = len(ids)
        context = torch.tensor(ids, dtype=torch.long)
        if self.use_cuda:
            context = context.cuda()
        context = context.unsqueeze(0)
        tokens = context
        way = 'GPT2'

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
            if t == max_t - 1:
                picked_tokens = torch.LongTensor([self.ix_EOS] * logits.shape[0]).view(-1, 1)
                if self.use_cuda:
                    picked_tokens = picked_tokens.cuda()
            else:
                picked_tokens = pick_tokens(prob)

            cand = []
            for i in range(picked_tokens.shape[0]):
                for j in range(picked_tokens.shape[1]):
                    ix = picked_tokens[i, j].item()
                    _sum_logP = sum_logP[i] + logP[i, ix].item()
                    cand.append((_sum_logP, i, j))

            if not cand:
                break
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
            
            if not cur:
                break
            tokens = torch.cat([torch.cat(cur, dim=0), torch.cat(nxt, dim=0).unsqueeze(-1)], dim=-1)

        finished = sorted(finished, reverse=True)
        ret = []
        for prob, seq in finished:
            hyp = self.tokenizer.decode(seq).strip()
            ret.append((way, prob, hyp))
            if len(ret) == beam:
                break
        return sorted(ret, reverse=True)


def play_lm():
    lm = LanguageModel()
    while True:
        cxt = input('\nCONTEXT:\t')
        if not cxt:
            break
        ret = lm.predict(cxt)
        for way, prob, hyp in ret:
            print('%s %.3f\t%s'%(way, prob, hyp.replace('\n',' ')))


if __name__ == "__main__":
    play_lm()