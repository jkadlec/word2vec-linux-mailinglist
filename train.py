import gensim
import json
import random

def train(userdata_file):
    random.seed()
    with open(userdata_file, 'r') as f:
        users = json.load(f)

    sentences = []
    for k, v in users.items():
        for words in v:
            sentences.append(gensim.models.doc2vec.TaggedDocument(words=words.split(), tags=[k]))
    model = gensim.models.Doc2Vec(size=100, window=4, min_count=15, workers=1,alpha=0.025, min_alpha=0.025)
    model.build_vocab(sentences)
    for epoch in range(10):
        model.train(random.choice(sentences) for _ in range(len(sentences)))
        model.alpha -= 0.002 # decrease the learning rate
        model.min_alpha = model.alpha # fix the learning rate, no deca
        model.train(random.choice(sentences) for _ in range(len(sentences)))
        print('epoch %d' % epoch)
    return model

if __name__ == '__main__':
    import sys
    model = train(sys.argv[1])
    model.save(sys.argv[2])
