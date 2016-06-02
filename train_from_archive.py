import os
import json
import re
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import json

_STOPSET = set(stopwords.words('english'))
_CHARS = set(['``' , "''", '.', ',', '(', ')', '...', ':', '-', '!', '?', ';', '\n', '\t', "'s", "'m", "'ll"])
_UNWANTED = _STOPSET.union(_CHARS)

_stemmer = SnowballStemmer("english")


def filter_message(message):
    message = message.lower()
    message = re.sub('<?https?:\/\/.*?( |>)', '', message)
    message = re.sub('[0-9,a-z]{20}', '', message) # commit SHAs
    message = re.sub('((Signed-off-by)|(Acked-by)).*', '', message, flags=re.DOTALL) # remove patches
    tokens = word_tokenize(message)
#    tokens = list(map(_stemmer.stem, tokens))
    tokens = list(filter(lambda w: w not in _UNWANTED, tokens))
    return ' '.join(tokens)


def load_files(mail_dir):

    files = (mail_dir + '/' + f for f in os.listdir(mail_dir))
    files_js = defaultdict(lambda: [])

    freqs = defaultdict(lambda: 0)
    counter = 0
    for filename in files:
        with open(filename, 'r') as f:
            file_js = json.load(f)
            if 'clean_body' in file_js:
                files_js[file_js['sender']].append(filter_message(file_js['clean_body']))
            freqs[file_js['sender']]+=1
            counter+=1
            if counter % 1000 == 0:
                print('.')

    old_count = len(files_js)
    files_js = {k:v for k, v in files_js.items() if freqs[k] > 30}
    print('filtered %d users' % (old_count - len(files_js)))
    return files_js

if __name__ == '__main__':
    import sys
    from subprocess import call
    
    files = load_files(sys.argv[1])
    with open(sys.argv[2], 'w') as f:
        json.dump(files, f)
    print('data cleaned')
    del files
    if len(sys.argv) == 4:
        call(['python3', 'train.py', sys.argv[2], sys.argv[3]])
