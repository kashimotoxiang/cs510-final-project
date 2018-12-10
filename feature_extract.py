import spacy
import lyx
import collections
from spacy.attrs import HEAD
import numpy as np
from sklearn.preprocessing import normalize
from benepar.spacy_plugin import BeneparComponent
POS = ("Tag", "-LRB-", "-RRB-", ",", ":", ".", "''", "\"\"", "#", "``", "$", "ADD", "AFX", "BES", "CC", "CD", "DT", "EX", "FW", "GW", "HVS", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD",
       "NFP", "NIL", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "_SP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "XX")

ENTITY = ("X", 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
          'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL')

DEP = ("acl",
       "acomp",
       "advcl",
       "advmod",
       "agent",
       "amod",
       "appos",
       "attr",
       "aux",
       "auxpass",
       "case",
       "cc",
       "ccomp",
       "compound",
       "csubj",
       "csubjpass",
       "dative",
       "dep",
       "det",
       "dobj",
       "expl",
       "intj",
       "iobj",
       "mark",
       "meta",
       "neg",
       "nmod",
       "npadvmod",
       "nsubj",
       "nsubjpass",
       "nummod",
       "oprd",
       "parataxis",
       "pcomp",
       "pobj",
       "poss",
       "preconj",
       "predet",
       "prep",
       "prt",
       "punct",
       "quantmod",
       "relcl",
       "xcomp",
       "conj",
       "cop",
       "nn",
       "nounmod",
       "npmod",
       "obj",
       "obl",
       "ROOT")

pos_mapper = {x: i for i, x in enumerate(POS)}
entity_mapper = {x: i for i, x in enumerate(ENTITY)}
dep_mapper = {x: i for i, x in enumerate(DEP)}

en_freq = lyx.io.load_pkl("en_freq")
en_freq = collections.defaultdict(lambda: 0, en_freq)
nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent("benepar_en"))


class Example():
    def __init__(self, text, feature, label):
        self.text = feature
        self.feature = feature
        self.label = label


class Features():
    def __init__(self, text):
        self.text = text
        self.rank = en_freq[text]
        self.length = None
        self.tag = None
        self.dep = None
        self.is_alpha = None
        self.is_digit = None
        self.is_title = None
        self.like_num = None
        self.is_lower = None
        self.is_upper = None
        self.is_currency = None
        self.is_punct = None
        self.is_stop = None
        self.is_oov = None
        self.ner = None
        self.offset = None
        self.dep_offset = None
        self.ischunk = None


def find_root(token, depth):
    if token.dep_ == 'ROOT':
        return depth
    parent = token.ancestors.next
    find_root(parent)


def feature_extract(sent):
    sent = sent.replace(")", "").replace("(", "(")
    doc = nlp(sent)
    features = []
    sent = list(doc.sents)[0]
    tree_str = sent._.parse_string
    deps = []
    depth = 0

    for i, c in enumerate(tree_str):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if tree_str[i-1] != ")":
                deps.append(depth)

    for token in doc:
        feature = Features(token.text)
        feature.length = len(token.text)
        feature.tag = token.tag_
        feature.dep = token.dep_
        feature.is_alpha = token.is_alpha
        feature.is_digit = token.is_digit
        feature.is_title = token.is_title
        feature.like_num = token.like_num
        feature.is_lower = token.is_lower
        feature.is_upper = token.is_upper
        feature.is_currency = token.is_currency
        feature.is_punct = token.is_punct
        feature.is_stop = token.is_stop
        feature.is_oov = token.is_oov
        feature.ner = "X"
        feature.vector_norm = token.vector_norm
        feature.offset = abs(token.head.i-token.i)
        feature.dep_offset = abs(token.head.i-token.i)
        features.append(feature)

    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            features[i].ner = ent.label_

    noun_chunks = list(doc.noun_chunks)
    for chunk in noun_chunks:
        for i in range(chunk.start, chunk.end):
            feature.ischunk = 1

    return features


def feature2vec(sentFeature):
    res = []
    for sent in sentFeature:
        sentVec = []
        for word in sent:
            posVec = [0] * len(pos_mapper)
            posVec[pos_mapper[word.tag]] = 1
            entityVec = [0] * len(entity_mapper)
            entityVec[entity_mapper[word.ner]] = 1
            depVec = [0]*len(dep_mapper)
            depVec[dep_mapper[word.dep]] = 1
            boolFeature = [word.is_oov, word.is_stop, word.is_currency, word.is_punct, word.is_upper,
                           word.is_lower, word.like_num, word.is_title, word.is_digit, word.is_alpha, word.ischunk]*1
            numFeaure = [word.length, word.dep_offset,
                         word.offset, word.vector_norm, word.rank]
            vec = boolFeature + posVec + entityVec + depVec + numFeaure
            sentVec.append(vec)
        res.append(sentVec)
    return res


def main():

    labels = lyx.io.read_all_lines("data/labels-remove-zeros.txt")[:100]
    labels = [np.fromstring(line, dtype=int, sep=' ')
              for line in labels]

    sentences = lyx.io.read_all_lines("data/sentences-remove-zeros.txt")[:100]

    # # sentences = [
    # #     "Given an array A of strings, find any smallest string that contains each string in A as a substring."]
    # # sentFeature = list(map(feature_extract, sentences))
    sentFeature = []
    for i, item in enumerate(sentences):
        tmp = feature_extract(item)
        sentFeature.append(tmp)
        print(i)

    lyx.io.save_pkl(sentFeature, "sentFeature")
    # sentFeature = lyx.io.load_pkl("sentFeature")
    sentFeatureVec = feature2vec(sentFeature)
    sentFeatureVec = np.array(sentFeatureVec)
    lyx.io.save_pkl(sentFeatureVec, "sentFeatureVec")

    exmaples = [Example(x, y, z)
                for x, y, z in zip(sentences, sentFeatureVec, labels)]
    lyx.io.save_pkl(exmaples, "exmaples")


if __name__ == "__main__":
    main()
