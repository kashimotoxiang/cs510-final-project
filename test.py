# %%

from BERT import sent2vec
import numpy as np

generator = sent2vec.VecGenerator()
# %%
sent = "That goes for your hands too, wash your hands frequently and always use chopsticks."
vecs = generator(sent)

vecs = np.array(vecs, np.float32)
np.resize(vecs, (1, 768))
import vector_search  # This import may take several minutes

# vecs = np.random.random((1, 768)).astype('float32')  # 这里随机生成一个向量来用
a = vector_search.search([vecs], 5)

print(a)
