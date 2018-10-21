# -*- coding: utf-8 -*-

import argparse
import logging
import os
import multiprocessing
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def run(infile, outfile):
    if not os.path.isfile(infile):
        print("not such file {}".format(infile))
        return

    outfile = os.path.abspath(outfile)
    root = os.path.dirname(outfile)
    os.makedirs(root, exist_ok=True)

    model = Word2Vec(LineSentence(infile), workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format(outfile, binary=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    args = parser.parse_args()
    run(args.infile, args.outfile)
