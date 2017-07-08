class TextModel:

    def predict(self, text):
        from numpy.random import choice
        probs = self.probs(text)
        result = probs.argmax()
        return chr(result)

    def probs(self, text):
        weights = self.weights(text)
        return weights / weights.sum()

    def sample(self, text):
        from numpy.random import choice
        probs = self.probs(text)
        result = choice(a=len(probs), p=probs)
        return chr(result)


class KerasTextModel(TextModel):

    def __init__(self, *, model):
        self.model = model

    def weights(self, text):
        from numpy import fromiter, int8
        text = fromiter((ord(_) for _ in text), dtype=int8)
        weights = self.model.predict(text.reshape([1, -1]))[0]
        # print(weights.sum())
        return weights


class TableTextModel(TextModel):

    def __init__(self, *, nkey=None, table):
        self.nkey = table.ndim - 1 if nkey is None else nkey
        self.table = table

    def weights(self, text):
        weights = self.table
        # print(self.table.shape)
        if self.nkey:
            keys = text[-self.nkey:]
            # print(keys)
            for key in keys:
                # print(key, ord(key))
                weights = weights[ord(key)]
        return weights


def build_network(*, seqs):
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import GRU
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.utils import to_categorical
    from numpy.random import choice
    model = Sequential()
    model.add(Embedding(input_dim=128, output_dim=2))
    model.add(GRU(return_sequences=True, units=32))
    model.add(GRU(activation='softmax', units=128))
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(decay=1e-8, lr=1e-2))
    # train = seqs[choice(a=len(seqs), replace=False, size=int(1e6))]
    train = seqs[:int(len(seqs) * 0.6)]
    train_x = train[:, :-1]
    train_y = to_categorical(num_classes=128, y=train[:, -1])
    print('about to train')
    model.fit(x=train_x, y=train_y, epochs=2)
    # Save the model.
    from datetime import datetime
    from os import makedirs
    from os.path import join
    now = datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')
    dir_name = join('notes', 'models')
    makedirs(dir_name, exist_ok=True)
    out_name = join(dir_name, f'seq-{time}.keras.h5')
    model.save(out_name)
    print(f'model saved to {out_name}')
    # for i in range(1000):
    #     model.train_on_batch(x=)
    return model


def gen_text(*, seed, tm):
    message = seed
    for _ in range(1000):
        message += tm.sample(message[-10:])
    print(message)


def hack_step(*, seq):
    from numpy import array, bincount, vstack
    # Model that always predicts most common.
    constant_accuracy = bincount(seq).max() / len(seq)
    # Model that predicts from single step.
    pairs = ngramify(n=2, seq=seq)
    counts = vstack(
        bincount(pairs[pairs[:, 0] == code, 1], minlength=128)
        for code in range(128))
    counts2 = counts
    counts_n1 = counts2
    pairs2 = pairs
    probs2 = counts.astype(float) / counts.sum(axis=1, keepdims=True)
    # Report.
    model = counts2.argmax(axis=1)
    predictions = model[pairs[:, 0]]
    accuracy = (predictions == pairs2[:, 1]).mean()
    print(f'baseline accuracy: {constant_accuracy} or {accuracy}')
    print(len(seq))
    # Look some at generation.
    tm = TableTextModel(table=probs2)
    gen_text(seed='Wha', tm=tm)
    # print(sum(probs), chr(probs.argmax()))
    # 3-grams.
    triples = ngramify(n=3, seq=seq)
    counts3 = []
    for start_code in range(128):
        pairs = triples[triples[:, 0] == start_code, 1:]
        counts = vstack(
            bincount(pairs[pairs[:, 0] == code, 1], minlength=128)
            for code in range(128))
        counts3.append(counts)
    counts3 = array(counts3)
    counts_n2 = counts3
    # print(counts3.shape)
    # Report.
    preds3 = counts3[triples[:, 0], triples[:, 1]].argmax(axis=1)
    acc3 = (preds3 == triples[:, 2]).mean()
    print(f'2-gram accuracy: {acc3}')
    # Generate.
    tm3 = TableTextModel(table=counts3.astype(float))
    gen_text(seed='Wha', tm=tm3)
    if False:
        # 4-grams.
        def incept():
            from collections import defaultdict
            return defaultdict(incept)
        grams4 = ngramify(n=4, seq=seq)
        counts4 = incept()
        total = 0
        for code0 in range(128):
            grams3 = grams4[grams4[:, 0] == code0, 1:]
            counts3 = incept()
            for code1 in range(128):
                grams2 = grams3[grams3[:, 0] == code1, 1:]
                counts2 = incept()
                for code2 in range(128):
                    grams1 = grams2[grams2[:, 0] == code2, 1]
                    if len(grams1):
                        counts1 = bincount(grams1)
                        total += (counts1 > 0).sum()
                        counts2[code2] = counts1
                if counts2:
                    counts3[code1] = counts2
            if counts3:
                counts4[code0] = counts3
        counts_n3 = counts4
        print(f'done counting 3-gram probs, total {total}')
        # Generate.
        tm4 = TableTextModel(nkey=3, table=counts4)
        gen_text(seed='Wha', tm=tm4)
        # Report.
        preds4 = []
        for gram4 in grams4:
            counts1 = counts4[gram4[0]][gram4[1]][gram4[2]]
            preds4.append(counts1.argmax())
        preds4 = array(preds4)
        acc4 = (preds4 == grams4[:, 3]).mean()
        print(f'3-gram accuracy: {acc4}')
    # Save the models.
    from datetime import datetime
    from json import dump
    from os import makedirs
    from os.path import join
    now = datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')
    dir_name = join('notes', 'models')
    makedirs(dir_name, exist_ok=True)
    name = join(dir_name, f'seq-{time}.ngram.json')
    if False:
        with open(name, 'w') as out_file:
            dump(
                default=lambda _: _.tolist(),
                fp=out_file,
                obj={'n1': counts_n1, 'n2': counts_n2, 'n3': counts_n3})


def infer(*, model, seqs):
    from numpy.random import choice
    for index in choice(a=len(seqs), replace=False, size=30):
        x = seqs[index][:-1]
        y = seqs[index][-1]
        probs = model.predict(x.reshape([1, -1]))[0]
        tops = probs.argsort()[::-1][:10]
        tops = ''.join(chr(_) for _ in tops)
        start = ''.join(chr(_) for _ in x)
        expected = chr(y)
        print(' -- '.join(repr(_) for _ in [start, expected, tops]))


def load_text(name):
    from numpy import fromiter, int8
    from unidecode import unidecode
    with open(name, encoding='utf8') as source_file:
        text = source_file.read()
    # Chop first off in case of byte order junk.
    text = text[1:]
    # Change to ascii.
    text = unidecode(text)
    # Unwrap newlines, presuming blank lines between paragraphs and no crs.
    text = (text.
        replace('\r', '').replace('\n\n', '\r').replace('\n', ' ').
        replace('\r', '\n'))
    # Make an array.
    text = fromiter((ord(_) for _ in text), dtype=int8)
    return text


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--source')
    args = parser.parse_args()
    text = load_text(args.source)
    if False:
        from numpy import bincount
        print(bincount(text))
    if True:
        # Show baselines.
        hack_step(seq=text)
        # return
    # Now work with a network.
    seqs = ngramify(seq=text, n=11)
    print(seqs.shape)
    if args.model:
        from keras.models import load_model
        model = load_model(args.model)
    else:
        model = build_network(seqs=seqs)
    infer(model=model, seqs=seqs)
    tm = KerasTextModel(model=model)
    gen_text(seed='Wha', tm=tm)


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
