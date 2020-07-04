#// Copyright (c) Microsoft Corporation.// Licensed under the MIT license. 

import numpy as np
from shared import alnum_only
import os


class ScorerRepetition:
    # measuring repetition penalty, proposed in https://arxiv.org/abs/2005.08365

    def predict(self, txts):
        scores = []
        for txt in txts:
            ww = []
            for w in alnum_only(txt).split():
                if w:
                    ww.append(w)
            if not ww:
                return 0
            rep = 1 - len(set(ww)) / len(ww)
            scores.append(- rep)
        return scores


class ScorerInfo:
    # measuring informativeness, proposed in https://arxiv.org/abs/2005.08365

    def __init__(self):
        fld = 'models/info'
        os.makedirs(fld, exist_ok=True)
        self.path = fld + '/common.txt'
    

    def load(self):
        self.w2rank = dict()
        for i, line in enumerate(open(self.path)):
            for w in line.strip('\n').split():
                self.w2rank[w] = i
        self.max_rank = i


    def score(self, w):
        rank = self.w2rank.get(w, self.max_rank)
        return rank/self.max_rank


    def train(self, path_corpus, max_n=1e6):
        from collections import Counter, defaultdict
        counter = Counter()
        n = 0
        for line in open(path_corpus, encoding='utf-8'):
            ww = alnum_only(line).split()
            for w in ww:
                if w:
                    counter[w] += 1
            n += 1
            if n == max_n:
                break
        
        freq_ww = defaultdict(list)
        for w, freq in counter.most_common():
            if freq < 5:
                break
            freq_ww[freq].append(w)

        lines = []
        for freq in sorted(list(freq_ww.keys()), reverse=True):
            lines.append(' '.join(freq_ww[freq]))
            
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        

    def predict(self, txts):
        scores = []
        for txt in txts:
            score = []
            ww = set(alnum_only(txt).split())
            for w in ww:
                if w:
                    score.append(self.score(w))
            scores.append(np.mean(score))
        return scores


class Ranker:

    def __init__(self, scorer_fwd=None, scorer_rvs=None, scorer_style=None):
        self.scorer_fwd = scorer_fwd
        self.scorer_rvs = scorer_rvs
        self.scorer_style = scorer_style
        
        self.scorer_rep = ScorerRepetition()
        self.scorer_info = ScorerInfo()
        self.scorer_info.load()


    def predict(self, cxt, hyps):
        info = self.scorer_info.predict(hyps)
        rep = self.scorer_rep.predict(hyps)
        if self.scorer_fwd is not None:
            prob_fwd = self.scorer_fwd.tf_prob(cxt, hyps)
        if self.scorer_rvs is not None:
            prob_rvs = self.scorer_rvs.rvs_prob(cxt, hyps)

        scored = []
        for i, hyp in enumerate(hyps):
            d = {
                'rep': rep[i],
                'info': info[i],
                }
            if self.scorer_fwd is not None:
                d['fwd'] = float(prob_fwd[i])
            if self.scorer_rvs is not None:
                d['rvs'] = float(prob_rvs[i])
            score = sum([d[k] for k in d])
            d['score'] = score + np.random.random() * 1e-8              # to avoid exactly the same score
            scored.append(d)
        return scored


def play_ranker():
    ranker = Ranker()
    while True:
        txt = input('\nTXT:\t')
        if not txt:
            break
        scored = ranker.predict('', [txt])[0]
        print(' '.join(['%s %.4f'%(k, scored[k]) for k in scored]))


if __name__ == "__main__":
    play_ranker()
    