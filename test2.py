
from BERT import sent2vec
import numpy as np

sent = "That goes for your hands too, wash your hands frequently and always use chopsticks."

from service.client import BertClient
bc = BertClient(ip='192.168.50.27')  # ip address of the GPU machine
vecs = bc.encode([sent])


import vector_search  # This import may take several minutes

# vecs = np.random.random((1, 768)).astype('float32')  # 这里随机生成一个向量来用
a = vector_search.search([vecs], 5)

print(a)
