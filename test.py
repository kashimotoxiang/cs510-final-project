# %%

from BERT import sent2vec
import numpy as np
from lyx.sql import DataBase
db = DataBase(database="cs510", host="127.0.0.1", user="root",
              password="123456")

generator = sent2vec.VecGenerator()
# %%
sent = "That goes for your hands too, wash your hands frequently and always use chopsticks."
vecs = generator(sent)

vecs = np.array(vecs, np.float32)
np.resize(vecs, (1, 768))
import vector_search  # This import may take several minutes

# vecs = np.random.random((1, 768)).astype('float32')  # 这里随机生成一个向量来用
a = vector_search.search([vecs], 5)
a = [int(x)+1 for x in a]
print(a)
res = [db.query("select sentences from sentences where QA_id=%d" % x)
       for x in a]
for item in res:
    print(item)
