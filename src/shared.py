import re

def get_api_key(name):
    for line in open('args/api.tsv'):
        ss = line.strip('\n').split('\t')
        if ss[0] == name:
            return ss[1:]
    return None


def alnum_only(s):
    return re.sub(r"[^a-z0-9]", ' ', s.lower())