import argparse
import copy
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize
from pdb import set_trace

import pexpect
import spacy

import lyx


TOK = None
# nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('en')


def _convert(text):
    if text == "ORG":
        return "ORGANIZATION"
    elif text == "DATA":
        return "NUMBER"
    else:
        return text


def process_dataset(data):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""

    for idx, line in enumerate(data):
        doc = nlp(line)

        # document tokenizations
        # document = [_convert(x) for x in c_tokens[idx]['words']]
        document = [token.text for token in doc]
        pos = [token.tag_ for token in doc]
        ner = []
        pre_end = 0
        for ent in doc.ents:
            label = _convert(ent.label_)
            ner = ner+(ent.start-pre_end) * ["O"]+[label]*(ent.end-ent.start)
            pre_end = ent.end

        ner = ner+(len(pos)-pre_end) * ["O"]
        yield {
            'document': document,
            'pos': pos,
            'ner': ner
        }


#####################
# parameters
# nlp = spacy.load('en_core_web_sm')
num_sents = 0
in_file = 'preprocessing/test.txt'
out_file = 'preprocessing/output_tmp.json'
dataset = lyx.io.read_all_lines(in_file)

#####################
# spacy
t0 = time.time()

print('Will write to file %s' % out_file)
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset):
        f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))


# 3. load the DrQA corenlp_tokenizer processed data
data = []
for line in open(out_file, 'r'):
    data.append(json.loads(line))

# do the truncation
for i in range(len(data)):
    # get case info
    doc_case = []
    for w in data[i]['document']:
        if w.isalpha():
            if w.islower():
                doc_case.append('L')
            else:
                doc_case.append('U')
        else:
            doc_case.append('L')
    data[i]['doc_case'] = doc_case

# append case, pos, ner, ansInd as features to context
out_file = 'test/output.txt'
with open(out_file, 'wb') as f:
    for i, ex in enumerate(data):
        print(i)
        line = u' '.join([ex['document'][idx].replace(' ', '').lower() + '￨' + ex['doc_case'][idx] + '￨' +
                          ex['pos'][idx] + '￨' + ex['ner'][idx] + '￨' + "-"
                          for idx in range(len(ex['document']))]).encode('utf-8').strip()
        f.write(line + u'\n'.encode('utf-8'))
f.close()
