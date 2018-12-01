import spacy
sent1 = "Examples pertinent to this crisis included: the adjustable-rate mortgage; the bundling of subprime mortgages into mortgage-backed securities (MBS) or collateralized debt obligations (CDO) for sale to investors, a type of securitization; and a form of credit insurance called credit default swaps (CDS)."
nlp = spacy.load('en')
doc1 = nlp(sent1)
tokens = [x for x in doc1]
pass
