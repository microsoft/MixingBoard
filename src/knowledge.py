#// Copyright (c) Microsoft Corporation.// Licensed under the MIT license. 

import pke, tagme, time
import numpy as np
import pdb, pickle, os, wikipedia, re, json, requests
from shared import get_api_key

from azure.cognitiveservices.search.websearch import WebSearchAPI
from azure.cognitiveservices.search.newssearch import NewsSearchAPI
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.search.websearch.models import SafeSearch


def extract_keyphrase(txt, n_max=5):
    try:
        extractor = pke.unsupervised.TopicRank()
        extractor.load_document(input=txt, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        results = extractor.get_n_best(n=n_max)
    except ValueError:
        return dict()

    ret = dict()
    for k, score in results:
        ret[k] = score
    return ret


class KnowledgeBase:
    # select the most relavant snippets from external knowledge source

    def __init__(self):
        self.fld = 'kb'
        bing_v7_key = get_api_key('bing_v7')[0]
        self.bing_news_client = NewsSearchAPI(CognitiveServicesCredentials(bing_v7_key))
        self.bing_web_client = WebSearchAPI(CognitiveServicesCredentials(bing_v7_key))
        

    def build_query(self, query, site=None, must_include=[]):
        if site is not None:
            query = query + ' site:%s'%site.strip()
        return query + ' ' + ' '.join(['"%s"'%w for w in must_include])


    def search_bing_news(self, query, site=None, must_include=[]):
        # call Bing news API (cognative serivices)
        # https://docs.microsoft.com/en-us/azure/cognitive-services/bing-news-search/news-sdk-python-quickstart
        url_snippet = []
        snippets = []
        news_result = self.bing_news_client.news.search(
            query=self.build_query(query, site, must_include), 
            market="en-us", count=10)
        for news in news_result.value:
            snippet = news.name + ' . ' + news.description
            snippets.append(snippet)
            url_snippet.append((news.url, snippet))
        return url_snippet

    
    def search_bing_web(self, query, site=None, must_include=[]):
        # https://docs.microsoft.com/en-us/azure/cognitive-services/bing-web-search/web-sdk-python-quickstart
        ord_url_snippet = []
        snippets = []
        t0 = time.time()
        query = self.build_query(query, site, must_include)
        web_data = self.bing_web_client.web.search(
            query=query,
            )
        if web_data.web_pages is None:
            return []
        for i, data in enumerate(web_data.web_pages.value):
            snippets.append(data.snippet)
            ord_url_snippet.append((i, data.url, data.snippet))
        return ord_url_snippet


    def pick_wiki_snippet(self, query, title):
        try:
            page = wikipedia.page(title)
        except:
            return None
        lines = page.content.split('\n')
        if len(lines) == 0:
            return False
        snippets = []
        qq = set(re.sub(r"[^a-z]", " ", query.lower()).split())
        for s in lines:
            s = s.strip('\n').strip()
            if s.startswith('=='):
                continue
            if len(s) > 30:
                ww = set(re.sub(r"[^a-z]", " ", s.lower()).split())
                score = len(ww & qq)/len(qq) + len(ww)/10
                snippets.append((score, s))
        snippets = sorted(snippets, reverse=True)
        cat = ''
        for _, snippet in snippets:
            cat += snippet
            if len(cat.split()) > 256:
                break
        return cat


    def rank_passage(self, query, ord_url_snippet, n_max=3):
        # a heuristic method by Xiang Gao, see Section 3.1 of https://arxiv.org/abs/2005.08365

        query_k = extract_keyphrase(query, n_max=n_max)
        set_q = dict()
        for k in query_k:
            set_q[k] = set(k.lower().split())
        n = len(ord_url_snippet)
        score_i = []
        for i in range(n):
            order, _, snippet = ord_url_snippet[i]
            ww = set(re.sub(r"[^a-z]", " ", snippet.lower()).split())
            overlap = 0
            for k in query_k:
                overlap += len(set_q[k] & ww)/len(set_q[k]) * query_k[k]
            score = overlap * 1./(order + 1)
            score_i.append((score, i))
        picked = pick_top(score_i)
        return [(ord_url_snippet[i][1], ord_url_snippet[i][2])  for _, i in picked[:min(len(picked), n_max)]]


    def predict(self, query, args=None):
        query = query.lower()
        if args is None:
            args = {'domain':'web', 'must_include':[]}
        domain = args['domain']

        if domain in ['web', 'cust']:
            site = args['cust'] if domain == 'cust' else None
            url_snippet = self.search_bing_web(query, site=site, must_include=args['must_include'])
            allowed = args.get('allowed')
            if allowed is not None:
                kept = []
                for url, snippet in url_snippet:
                    if url in allowed:
                        kept.append([url, snippet])
                url_snippet = kept[:]
        elif domain == 'news':
            url_snippet = self.search_bing_news(query, must_include=args['must_include'])
        elif domain == 'user':
            lines = args['txt_kb'].split('\n')
            url_snippet = []
            window = 1
            for i in range(len(lines) - window + 1):
                snippet = ' '.join(lines[i:i+window]).strip()
                if len(snippet) > 0:
                    url_snippet.append(['user', snippet])
        else:
            raise ValueError
        ranked = self.rank_passage(query, url_snippet)
        return ranked


def pick_top(score_v, crit=0.1):
    if len(score_v) == 0:
        return []
    s = sorted(score_v, reverse=True)
    max_score = s[0][0]
    picked = []
    for score, v in s:
        if score < max_score * crit:
            break
        picked.append((score, v))
    return picked



if __name__ == "__main__":
    kb = KnowledgeBase()
    while True:
        print('\n(empty query to exit)')
        query = input('QUERY:\t')
        if not query:
            break
        url_snippets = kb.predict(query)
        for url, snippet in url_snippets:
            print('\nURL:\t%s'%url)
            print('TXT:\t%s'%snippet)
