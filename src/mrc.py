#// Copyright (c) Microsoft Corporation.// Licensed under the MIT license. 

from allennlp.predictors.predictor import Predictor

class BidafQA:
    def __init__(self):
        self.model = Predictor.from_path('models/BiDAF/bidaf.tar.gz')

    def predict(self, query, passage):
        ret = self.model.predict(passage=passage, question=query)
        span_str = ret['best_span_str']
        span_prob = ret['span_start_probs'][ret['best_span'][0]] * ret['span_end_probs'][ret['best_span'][1]]
        return [('Bidaf', span_prob, span_str)]
        
def play_mrc():
    model = BidafQA()
    while True:
        q = input('QUERY:\t')
        p = input('PASSAGE:\t')
        ret = model.predict(q, p)
        for way, prob, ans in ret:
            print('%s %.3f\t%s'%(way, prob, ans))

if __name__ == "__main__":
    play_mrc()
