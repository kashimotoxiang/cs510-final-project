import lyx
from service.client import BertClient
bc = BertClient(ip='127.0.0.1')  # ip address of the GPU machine
lines = lyx.io.read_all_lines("data/extend_mask_input.txt")
input = [x.split("\t")[1] for x in lines if len(x) != 0]
res = bc.encode(input)
lyx.io.save_pkl(res, "res")
