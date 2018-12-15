import collections
import lyx
subsent_mapper = {}
extend_mask_input = lyx.io.read_all_lines("extend_mask_input.txt")
all_sentences = lyx.io.read_all_lines("All sentences.txt ")
question_ids = ...
for line_number, line in enumerate(extend_mask_input):
    idx = line.split("\t")[0]
    if idx in question_ids:
        subsent_mapper[line_number] = idx

res_id = vectorsearch(...)
question = all_sentences[subsent_mapper[res_id]]
