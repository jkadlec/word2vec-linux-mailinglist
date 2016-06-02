import gensim

import numpy as np

from sklearn.manifold import TSNE

# Random state.
RS = 20150101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

import gensim

from collections import defaultdict
import json

f2s = gensim.matutils.full2sparse
cs = gensim.matutils.cossim


def _get_highest_similarities(m, doctag):
    doctag_vector = f2s(m.docvecs[doctag])
    sims = [{'word':k, 'score':cs(doctag_vector, f2s(m[k]))} for k in m.vocab.keys()]
    sims = sorted(sims, key=lambda s: s['score'], reverse=True)
    return sims[:500]


def _get_all_similarities(m, n):
    all_sims = defaultdict(lambda: 0.0)
    for doctag_name in list(m.docvecs.doctags.keys())[:n]:
        doctag_vector = f2s(m.docvecs[doctag_name])
        sims = [{'word':k, 'score':cs(doctag_vector, f2s(m[k]))} for k in m.vocab.keys()]
        for s in sims:
            all_sims[s['word']] += s['score']
    all_sims = sorted([{'word':k, 'score':v} for k, v in all_sims.items()], key=lambda s: s['score'], reverse=True)[0:500]
    return all_sims


def _load_and_filter_model(modelpath, vocab_cutoff):
    m = gensim.models.Doc2Vec.load(modelpath)
    vocab_dict = [{'word':k, 'count':v.count} for k, v in m.vocab.items()]
    vocab_top_n = sorted(filter(lambda w: w['word'].isalnum(), vocab_dict), key=lambda v: v['count'], reverse=True)[0:vocab_cutoff]

    return m, vocab_top_n


def main(modelpath, cutoff, plot_tags, cluster_count, id_map):
    if id_map is not None:
        with open(id_map, 'r') as f:
            id_map = json.load(f)
    model, vocab = _load_and_filter_model(modelpath, cutoff)
    if plot_tags:
        labels = [w['word'] for w in vocab]
        x = np.vstack(model[w['word']] for w in vocab)
    else:
        if id_map:
            labels = [id_map[k] for k in model.docvecs.doctags.keys()]
        else:
            labels = [k for k in model.docvecs.doctags.keys()]

        x = np.vstack(model.docvecs[u] for u in model.docvecs.doctags.keys())

    proj = TSNE(random_state=RS).fit_transform(x)

    fig, ax = plt.subplots()
    plot_x = proj[:,0]
    plot_y = proj[:,1]
    ax.scatter(plot_x, plot_y)

    for i, label in enumerate(labels):
        ax.annotate(label, (plot_x[i],plot_y[i]))

    plt.show()


if __name__ == '__main__':
    import sys

    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] == 'tags', int(sys.argv[4]), sys.argv[5] if sys.argv[5] != 'None' else None)
