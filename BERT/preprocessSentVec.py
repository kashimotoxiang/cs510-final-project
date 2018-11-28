import lyx
from service.client import BertClient
bc = BertClient(ip='127.0.0.1')  # ip address of the GPU machine
lines = lyx.io.read_all_lines("/root/cs510/bert-master/output.txt")
res = bc.encode(lines)
lyx.io.save_pkl(res, "res")
