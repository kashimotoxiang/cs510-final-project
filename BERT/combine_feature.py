import lyx
import glob
import os
from .dataStructure import outputFeatures


def main():
    files = glob.glob('/root/HHD/tmp/*.pkl')
    files_tuple = [(int(os.path.splitext(os.path.basename(x))[0]), x)
                   for x in files]
    files_tuple = sorted(files_tuple, key=lambda x: x[0])
    print('\n\nfiles_tuple:')
    print(files_tuple)
    res = []
    for id, file in files_tuple:
        print('\n\nid:')
        print(id)
        tmp = lyx.io.load_pkl(file.split(".")[0])
        res += [x.sentVec for x in tmp]
    lyx.io.save_pkl(res, "sentVec")


if __name__ == "__main__":
    pass
