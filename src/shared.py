

def get_api_key(name):
    for line in open('args/api.tsv'):
        ss = line.strip('\n').split('\t')
        if ss[0] == name:
            return ss[1:]
    return None