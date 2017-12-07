import cPickle
import numpy as np
import gensim


def preEmbed(vocab):
    f = open('glove.840B.300d.txt')
    embed_text = ''
    pre_embed = {}
    rep_num = 0
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab:
            rep_num += 1
            embed = np.asarray(values[1:], dtype = 'float32')
            widx = vocab[word]
            pre_embed[widx] = embed
            embed_text += line
    f.close()
    print '{} words has pre-trained embeddings!'.format(rep_num)
    # cPickle.dump(embed_text, open('pre_embed.txt', 'wb'))
    return pre_embed

def readEmbed(vocab):
    pre_embed = cPickle.load(open('pre_embed.txt','r'))
    embed = pre_embed.split('\n')
    pre_embed = {}
    num = 0
    for e in embed:
        values = e.split()
        word = values[0]
        if word in vocab:
            embed = np.asarray(values[1:], dtype='float32')
            widx = vocab[word]
            pre_embed[widx] = embed
            num+=1
    print '{} words are pre embedded'.format(num)
    return pre_embed


def w2v(vocab):
    #model = gensim.models.Word2Vec.load('./all.model')
    model = gensim.models.Word2Vec.load('./clean2.model')
    pre_embed = {}
    single = [0.0] * 300
    em = [single] * 2
    rep_num = 0
    for w in model.vocab:
        if w in vocab:
            rep_num += 1
            embed = model[w]
            widx = vocab[w]
            pre_embed[widx] = embed
            em.append(embed.tolist())
    print '{} words are pre embedded'.format(rep_num)
    return pre_embed, em

def w2v1(vocab):
    #model = gensim.models.Word2Vec.load('./all.model')
    model = gensim.models.Word2Vec.load('./clean21.model')
    pre_embed = {}
    single = [0.0] * 300
    em = [single] * 2
    rep_num = 0
    for w in model.vocab:
        if w in vocab:
            rep_num += 1
            embed = model[w]
            widx = vocab[w]
            pre_embed[widx] = embed
            em.append(embed.tolist())
    print '{} words are pre embedded'.format(rep_num)
    return pre_embed, em