#// Copyright (c) Microsoft Corporation.// Licensed under the MIT license. 

import json, subprocess, torch, os, time, pdb
import numpy as np
from knowledge import KnowledgeBase
import base64
from flask import Flask, request, render_template
from lm import LanguageModel
from grounded import ContentTransfer
from flask_restful import Resource, Api
from ranker import Ranker
score_names = ['fwd', 'rep', 'info', 'score']


class DialogBackend:
    def get_query(self, context):
        return context.split('. ')[-1]


class DialogBackendLocal(DialogBackend):

    def __init__(self):
        super().__init__()
        
        self.model_lm = LanguageModel()
        self.model_ct = ContentTransfer()
        self.kb = KnowledgeBase()
        self.ranker = Ranker(self.model_lm)
        self.local = True


    def predict(self, context, max_n=1):
        print('backend running, context = %s'%context)
        query = self.get_query(context)

        # get results from different models
        results = self.model_lm.predict(context)

        passages = []
        url_snippet = []
        for line in open('args/kb_sites.txt', encoding='utf-8'):
            cust = line.strip('\n')
            kb_args = {'domain': 'cust', 'cust': cust, 'must_include':[]}
            url_snippet.append(self.kb.predict(query, args=kb_args)[0])
            passage = ' ... '.join([snippet for _, snippet in url_snippet])
            passages.append((passage, query))

        for passage, kb_query in passages:
            results += self.model_ct.predict(kb_query, passage)

        # rank hyps from different models

        hyps = [hyp for _, _, hyp in results]
        scored = self.ranker.predict(context, hyps)
        ret = []
        for i, d in enumerate(scored):
            d['way'], _, d['hyp'] = results[i]
            ret.append((d['score'], d))
        ranked = [d for _, d in sorted(ret, reverse=True)]
        if max_n > 0:
            ranked = ranked[:min(len(ranked), max_n)]
        return ranked, url_snippet
        


class DialogBackendRemote(DialogBackend):

    def __init__(self, host):
        super().__init__()
        self.local = False
        self.host = host
        
    def predict(self, context):
        cmd = 'curl http://%s/ -d "context=%s" -X GET'%(self.host, context)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output, error = process.communicate()
        if error is not None:
            print(error)
            return [], []
        ret = json.loads(output.decode())
        return ret['responses'], ret['passages']
        

def cmd_demo(backend):
    while True:
        src = input('\nUSER: ')
        if len(src) == 0:
            break
        with torch.no_grad():
            ranked, url_passages = backend.predict(src)
        for url, passage in url_passages:
            print(url)
            print(passage)
            print()

        for d in ranked:
            ss = []
            for k in d:
                if k not in ['way', 'hyp']:
                    ss.append('%s %.3f'%(k, d[k]))
            line =  '\t'.join([' '.join(ss), d['way'], d['hyp'].replace('\n',' ')])
            print(line)



def web_demo(backend, port=5000):
    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    def root():
        if request.method == 'POST':
            context = request.form['inp_cxt']
            with torch.no_grad():
                dd_hyp, url_snippet = backend.predict(context)

            hyps = []
            for d in dd_hyp:
                score = ['%.2f'%d.get(k, np.nan) for k in score_names]
                hyps.append([score, d['way'], d['hyp'], d.get('hyp_en','')])
        
        else:
            context = ''
            url_snippet = []
            hyps = []
        
        max_len = 30
        passages = []
        for url, snippet in url_snippet:
            url_display = url.replace('http:','').replace('https:','').strip('/')
            url_display = url_display.replace('en.wikipedia.org','')
            if len(url_display) > max_len:
                url_display = url_display[:max_len] + '...'
            passages.append([url_display, url, snippet])
        
        html = render_template('doc_gen.html', 
                score_header=score_names,
                passages=passages, hyps=hyps, 
                cxt_val=context,
                )
        
        html = html.replace('value="1"','checked')
        html = html.replace('value="0"','')
        return html

    app.run(host='0.0.0.0', port=port)


class ApiResource(Resource):
    def __init__(self, backend):
        self.backend = backend
        
    def get(self):
        context = request.form['context']
        with torch.no_grad():
            dd_hyp, url_snippet = self.backend.predict(
                context,
                )
        
        ret = {
            # parsed input ----
            'context': context,
            'passages': url_snippet,
            'responses': dd_hyp,
        }
        return ret



def restful_api_demo(backend, port=5000):
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(ApiResource, '/', 
        resource_class_kwargs={'backend': backend})
    app.run(host='0.0.0.0', port=port)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="`cmd`, `api`, or `web`")
    parser.add_argument('--remote', type=str, default='')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    if args.remote:
        backend = DialogBackendRemote(args.remote)
    else:
        backend = DialogBackendLocal()
    if args.mode == 'cmd':
        cmd_demo(backend)
    elif args.mode == 'web':
        web_demo(backend, port=args.port)
    elif args.mode == 'api':
        restful_api_demo(backend, port=args.port)