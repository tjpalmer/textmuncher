def build_network(*, seqs):
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import GRU
    from keras.models import Sequential
    from keras.utils import to_categorical
    from numpy.random import choice
    model = Sequential()
    model.add(Embedding(input_dim=128, output_dim=4))
    model.add(GRU(activation='softmax', units=128))
    model.compile(
        loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    train = seqs[choice(a=len(seqs), replace=True, size=int(1e6))]
    train_x = train[:, :-1]
    train_y = to_categorical(num_classes=128, y=train[:, -1])
    print('about to train')
    model.fit(x=train_x, y=train_y, epochs=1)
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
    text = unidecode(text)
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
    seqs = ngramify(seq=text, n=11)
    print(seqs.shape)
    if args.model:
        from keras.models import load_model
        model = load_model(args.model)
    else:
        model = build_network(seqs=seqs)
    infer(model=model, seqs=seqs)


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
