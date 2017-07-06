def load_text(name):
    from numpy import fromiter, int8
    from unidecode import unidecode
    with open(name, encoding='utf8') as source_file:
        text = source_file.read()
    # Chop first off in case of byte order junk.
    text = text[1:]
    text = unidecode(text)
    text = fromiter((ord(_) for _ in text), dtype=int8)
    return text

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--source')
    args = parser.parse_args()
    text = load_text(args.source)
    from numpy import bincount
    # print(bincount(text))
    # return
    seqs = ngramify(seq=text, n=50)
    print(seqs.shape)

def ngramify(*, seq, n):
    from numpy import arange
    count = len(seq) - n + 1
    if count < 1:
        ngrams = seq[:0]
    else:
        ngrams = seq[arange(n)[None, :] + arange(count)[:, None]]
    return ngrams

if __name__ == '__main__':
    main()
