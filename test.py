# %%

from BERT import sent2vec
import vector_search  # This import may take several minutes
import numpy as np

import mysql.connector
import math

MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWD = '123456'
MYSQL_DB = 'cs510'

cnx = mysql.connector.connect(
    host=MYSQL_HOST, user=MYSQL_USER, passwd=MYSQL_PASSWD, database=MYSQL_DB)
cursor = cnx.cursor()
cursor.execute('select word, count from words')
words = cursor.fetchall()



generator = sent2vec.VecGenerator()
# %%
sent = "That goes for your hands too, wash your hands frequently and always use chopsticks."
vecs = generator(sent)
vecs = np.array(vecs, np.float32)
np.resize(vecs, (1, 768))
# vecs = np.random.random((1, 768)).astype('float32')  # 这里随机生成一个向量来用
a = vector_search.search([vecs], 5)

print(a)
