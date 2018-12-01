import spacy
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
sent1 = "On L2-normalized data, this function is equivalent to linear_kernel."
sent2 = "Examples pertinent to this crisis included: the adjustable-rate mortgage"
nlp = spacy.load('en')
doc1 = nlp(sent1)
doc2 = nlp(sent2)
res = cosine_similarity([doc1.vector, doc2.vector])
print('\n\nres:')
print(res)
pass
