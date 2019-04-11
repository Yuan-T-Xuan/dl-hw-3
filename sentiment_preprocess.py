import re
import pickle

def create_vocab(sourcefile_path, output_path):
    f = open(sourcefile_path)
    lines = f.read().split("\n")[:-1]
    f.close()
    vocab = dict()
    for line in lines:
        alpha_only = re.sub("[^a-z ]", "", line.lower())
        for word in alpha_only.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    #print(vocab)
    outf = open(output_path, 'wb')
    pickle.dump(vocab, outf)
    outf.close()

