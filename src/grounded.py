#// Copyright (c) Microsoft Corporation.// Licensed under the MIT license. 

from cmr.process_raw_data import filter_query, filter_fact
from cmr.batcher import load_meta, prepare_batch_data
from cmr.model import DocReaderModel
import json, os, torch
import numpy as np
from todo import pick_tokens


class JsonConfig:
    def __init__(self, path):
        d = json.loads(open(path, encoding='utf-8').readline())
        for k in d:
            setattr(self, k, d[k])


class ConversingByReading:
    # ref: https://github.com/qkaren/converse_reading_cmr

    def __init__(self, use_cuda=True):
        args = JsonConfig('models/cmr/args.json')
        self.embedding, self.opt, self.vocab = load_meta(vars(args), args.meta)
        self.opt['skip_tokens'] = self.get_skip_tokens(self.opt["skip_tokens_file"])
        self.opt['skip_tokens_first'] = self.get_skip_tokens(self.opt["skip_tokens_first_file"])
        self.state_dict = torch.load(args.model_dir)["state_dict"]
        self.model = DocReaderModel(self.opt, self.embedding, self.state_dict)
        self.model.setup_eval_embed(self.embedding)
        if use_cuda:
            self.model.cuda()

    def get_skip_tokens(self, path):
        skip_tokens = None
        if path and os.path.isfile(path):
            skip_tokens = []
            with open(path, 'r') as f:
                for word in f:
                    word = word.strip().rstrip('\n')
                    try:
                        skip_tokens.append(self.vocab[word])
                    except:
                        print("Token %s not present in dictionary" % word)
        return skip_tokens

    def predict(self, context, passage, top_k=2, verbose=False):
        data = [{'query': context, 'fact': passage}]

        def pred2words(prediction, vocab):
            EOS_token = 3
            outputs = []
            for pred in prediction:
                new_pred = pred
                for i, x in enumerate(pred):
                    if int(x) == EOS_token:
                        new_pred = pred[:i]
                        break
                outputs.append(' '.join([vocab[int(x)] for x in new_pred]))
            return outputs

        processed_data = prepare_batch_data([self.preprocess_data(x) for x in data], ground_truth=False)
        logPs, predictions = self.model.predict(processed_data, pick_tokens=pick_tokens)
        pred_word = pred2words(predictions, self.vocab)
        hyps = [np.asarray(x, dtype=np.str).tolist() for x in pred_word]
        return [('CMR', np.exp(logPs[i]), hyps[i]) for i in range(len(hyps))]


    def preprocess_data(self, sample, q_cutoff=30, doc_cutoff=500):
        def tok_func(toks):
            return [self.vocab[w] for w in toks]

        fea_dict = {}

        query_tokend = filter_query(sample['query'].strip(), max_len=q_cutoff).split()
        doc_tokend = filter_fact(sample['fact'].strip()).split()
        if len(doc_tokend) > doc_cutoff:
            doc_tokend = doc_tokend[:doc_cutoff] + ['<TRNC>']

        # TODO
        fea_dict['query_tok'] = tok_func(query_tokend)
        fea_dict['query_pos'] = []
        fea_dict['query_ner'] = []

        fea_dict['doc_tok'] = tok_func(doc_tokend)
        fea_dict['doc_pos'] = []
        fea_dict['doc_ner'] = []
        fea_dict['doc_fea'] = ''

        if len(fea_dict['query_tok']) == 0:
            fea_dict['query_tok'] = [0]
        if len(fea_dict['doc_tok']) == 0:
            fea_dict['doc_tok'] = [0]

        return fea_dict


class OptionContentTransfer:
    def __init__(self):
        self.path_model = 'models/crg/crg_model.pt'
        self.path_tokenizer = 'models/crg/bpeM.model'
        self.beam_size = 5
        self.batch_size = 1
        self.max_sent_length = 100
        self.replace_unk = True
        self.verbose = False
        self.n_best = 1
        self.cuda = True


import onmt
class ContentTransfer:
    # ref: https://github.com/shrimai/Towards-Content-Transfer-through-Grounded-Text-Generation

    def __init__(self):
        import sentencepiece as spm
        opt = OptionContentTransfer()
        self.model = onmt.Translator(opt)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(opt.path_tokenizer)

    def encode(self, s):
        return ' '.join(self.tokenizer.EncodeAsPieces(s.lower()))

    def decode(self, s):
        return self.tokenizer.DecodePieces(s).replace(chr(92),' ')

    def predict(self, query, passage, min_score_article=0.3, min_score_passage=0.1, verbose=False):
        cxt = self.encode(query)
        src = self.encode(passage)
        srcBatch = [src.split()]
        cxtBatch = [cxt.split()]
        hyp, _, _ = self.model.translate(srcBatch, cxtBatch, None)
        hyp = self.decode(hyp[0][0]).split('|')[0].strip()
        return [('CT', 0, hyp)]


        
def play_grounded(which):
    if which == 'cmr':
        model = ConversingByReading()
    elif which == 'ct':
        model = ContentTransfer()

    while True:
        cxt = input('\nCONTEXT:\t')
        if not cxt:
            break
        passage = input('\nPASSAGE:\t')
        if not passage:
            break
        ret = model.predict(cxt, passage)
        for way, prob, hyp in ret:
            print('%s %.3f\t%s'%(way, prob, hyp))


if __name__ == "__main__":
    import sys
    play_grounded(sys.argv[1])