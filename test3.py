from service.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
bc = BertClient(ip='192.168.50.27')  # ip address of the GPU machine
sent = ["That goes for your hands too, wash your hands frequently and always use chopsticks.",
        "wash your hands frequently and always use chopsticks."]
vecs = bc.encode(sent)
res = cosine_similarity([vecs[0], vecs[1]])
print('\n\nres:')
print(res)
