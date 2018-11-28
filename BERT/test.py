import lyx
import glob
import os
from dataStructure import outputFeatures


def main():
    tmp = lyx.io.load_pkl("BERT/1")
    res = [x.tokenVec[0][0] for x in tmp]
    lyx.io.save_pkl(res, "sentVec")


if __name__ == "__main__":
    main()
