from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize
from pdb import set_trace

import spacy

import lyx
nlp = spacy.load('en_core_web_lg')
# nlp = spacy.load('en')


#####################
# parameters
# nlp = spacy.load('en_core_web_sm')
in_file = 'preprocessing/test.txt'
out_file = 'test/output1.txt'

dataset = lyx.io.read_all_lines(in_file)


def _convert(text):
    if text == "ORG":
        return "WHAT ORGANIZATION"
    elif text == "DATE":
        return "WHEN"
    elif text == "PERSON":
        return "WHO"
    elif text == "LOC":
        return "WHERE"
    else:
        return "WHAT "+text


#####################
# spacy
res = []
for idx, line in enumerate(dataset):
    doc = nlp(line)

    # document tokenizations
    # document = [_convert(x) for x in c_tokens[idx]['words']]
    doc_str = doc.text
    res.append(str(idx)+"\t"+doc_str)
    tokens = [token.text for token in doc]
    for ent in doc.ents:
        tmp = tokens[0:ent.start]+[_convert(ent.label_)]+tokens[ent.end:]
        res.append(str(idx)+"\t"+" ".join(tmp))
res = "\n".join(res)
lyx.io.write_all(res, out_file)
