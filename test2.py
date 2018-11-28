
from BERT import sent2vec
import numpy as np
from lyx.sql import DataBase
db = DataBase(database="cs510", host="127.0.0.1", user="root",
              password="123456")

sent = "you're on the right track in understanding that he's frustrated."

from service.client import BertClient
bc = BertClient(ip='192.168.50.27')  # ip address of the GPU machine
vecs = bc.encode([sent])


import vector_search  # This import may take several minutes

# vecs = np.random.random((1, 768)).astype('float32')  # 这里随机生成一个向量来用
result_sentences, result_similarities = vector_search.search(vecs, 10)
a = [int(x)+1 for x in result_sentences[0]]
res = [db.query("select sentences from sentences where id=%d" % x)
       for x in a]

for i in range(len(res)):
    print(a[i])
    print(res[i])
    print(result_similarities[0][i])
    print()
