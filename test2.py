
from BERT import sent2vec
import numpy as np
from lyx.sql import DataBase
db = DataBase(database="cs510", host="127.0.0.1", user="root",
              password="xiangzaili1995")

sent = "That goes for your hands too, wash your hands frequently and always use chopsticks."

from service.client import BertClient
bc = BertClient(ip='192.168.50.27')  # ip address of the GPU machine
vecs = bc.encode([sent])


import vector_search  # This import may take several minutes

# vecs = np.random.random((1, 768)).astype('float32')  # 这里随机生成一个向量来用
a = vector_search.search(vecs, 5)
a = [int(x)+1 for x in a[0]]
print(a)

res = [db.query("select sentences from sentences where id=%d" % x)
       for x in a]
for item in res:
    print(item)
