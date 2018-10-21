# -*- coding: utf-8 -*-

import argparse
import os
import multiprocessing
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import progressbar
from gensim.corpora import WikiCorpus
from hanziconv import HanziConv
import jieba


def read_word(infile, inqueue, interrupt):
    wiki = WikiCorpus(infile, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        inqueue.put(text)
        if interrupt is not None:
            interrupt -= 1
            if interrupt <= 0:
                break
    return


def process(inqueue, outqueue):
    while True:
        line = inqueue.get()
        if line is None:
            break
        words = [HanziConv.toSimplified(w) for w in line]
        words = [w2 for w1 in words for w2 in jieba.cut(w1, cut_all=False)if len(w2) > 1]
        words = [w.encode('utf-8') for w in words]
        text = b' '.join(words)
        outqueue.put(text)
    return


def write_word(outfile, outqueue):
    with open(outfile, 'wb') as f:
        while True:
            line = outqueue.get()
            if line is None:
                break
            f.writelines([line, b'\n'])
    return


def run2(infile, outfile, interrupt):
    if not os.path.isfile(infile):
        print("not such file {}".format(infile))
        return False

    outfile = os.path.abspath(outfile)
    path = os.path.dirname(outfile)
    os.makedirs(path, exist_ok=True)

    buf = 4096
    inqueue = multiprocessing.Queue(maxsize=buf)
    outqueue = multiprocessing.Queue(maxsize=buf)

    reader = multiprocessing.Process(target=read_word, args=(infile, inqueue, interrupt))
    reader.start()

    cleaners = [multiprocessing.Process(target=process, args=(inqueue, outqueue))
                for _ in range(multiprocessing.cpu_count())]
    for cleaner in cleaners:
        cleaner.start()

    writer = multiprocessing.Process(target=write_word, args=(outfile, outqueue))
    writer.start()

    wait = 1
    while reader.is_alive():
        reader.join(wait)
    while cleaners:
        for _ in cleaners:
            inqueue.put(None, timeout=wait)
        stop_cleaners = []
        for cleaner in cleaners:
            cleaner.join(wait)
            if not cleaner.is_alive():
                stop_cleaners.append(cleaner)
        for cleaner in stop_cleaners:
            cleaners.remove(cleaner)
    while writer.is_alive():
        outqueue.put(None, timeout=wait)
        writer.join(wait)


def run(infile, outfile, interrupt):
    if not os.path.isfile(infile):
        print("not such file {}".format(infile))
        return

    outfile = os.path.abspath(outfile)
    root = os.path.dirname(outfile)
    os.makedirs(root, exist_ok=True)

    wiki = WikiCorpus(infile, lemmatize=False, dictionary={})
    with open(outfile, 'wb') as f:
        i = 0
        bar = progressbar.ProgressBar(maxval=progressbar.UnknownLength,
                                      widgets=[progressbar.widgets.Counter(), '', progressbar.widgets.Bar()])
        bar.start()
        for text in wiki.get_texts():
            text = [w.encode('utf-8') for w in text]
            f.writelines([b' '.join(text), b'\n'])
            i += 1
            bar.update(i)
            if interrupt and bar.currval >= interrupt:
                break
        bar.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--interrupt', type=int, required=False, default=None)
    args = parser.parse_args()

    # run(args.infile, args.outfile, args.interrupt)
    run2(args.infile, args.outfile, args.interrupt)
