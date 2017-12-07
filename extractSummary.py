"""
Extractive summary:
This model is to extract top-4 most similar sentences from News according to the
prediction results.
"""
import os
import re
import numpy as np
import cPickle
from sklearn.metrics.pairwise import cosine_similarity
from pythonrouge.pythonrouge import Pythonrouge
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf

def index(words, vocab, vocab_inv, num):
    text = []
    #words = words[2:-1].strip()
    words = re.sub(r'UNK', '', words)
    words = words.strip()
    for word in words.split():
        if word not in vocab and word.strip() is not 'RT' and word.strip() is not 'rt':
            # import pdb; pdb.set_trace()
            vocab[word] = num
            vocab_inv[num] = word
            num = num + 1
        text.append(vocab[word])
    return text, num

def loopDir(file_dir, dict, num, vocab, vocab_inv):
    for filename in sorted(os.listdir(file_dir)):  #, key=int
        abs_dir = os.path.join(file_dir, filename)
        print 'Processing ', filename
        if os.path.isfile(abs_dir):
            file_in = open(abs_dir, 'r')
            item = cPickle.load(file_in)
            items = item.split('\n')
            con = []
            for line in items:
                text, num = index(line, vocab, vocab_inv, num)
                con.append(text)
            dict.append(con)
    return num


def loopGRD(file_dir, dict, num, vocab, vocab_inv):
    for filename in sorted(os.listdir(file_dir)):  #, key=int
        abs_dir = os.path.join(file_dir, filename)
        print 'Processing ', filename
        # import pdb;
        if os.path.isfile(abs_dir):
            file_in = open(abs_dir, 'r')
            item = cPickle.load(file_in)
            items = item.split('\n')
            con = []
            # pdb.set_trace()
            for line in items:
                text, num = index(line, vocab, vocab_inv, num)
                con.append(text)
            dict.append(con)
            # pdb.set_trace()
    return num

def loop_beam(file_dir, dict, num, vocab, vocab_inv):
    for filename in sorted(os.listdir(file_dir)):  #, key=int
    #for filename in os.listdir(file_dir):
        abs_dir = os.path.join(file_dir, filename)
        print 'Processing ', filename
        if os.path.isfile(abs_dir):
            file_in = open(abs_dir, 'r')
            item = cPickle.load(file_in)
            items = item.split('\n')
            # import pdb; pdb.set_trace()
            new_con = []
            con = []
            for line in items:
                text, num = index(line, vocab, vocab_inv, num)
                con += text
                # con.append(text)
            new_con.append(con)
            dict.append(new_con)
    return num


def loopClu(pred_dir, predict, summary, num, vocab, vocab_inv, top):
    ord = 0
    for filename in sorted(os.listdir(pred_dir)):  #  , key=int
        abs_dir = os.path.join(pred_dir, filename)
        print 'Processing ', filename
        if os.path.isfile(abs_dir):
            file_in = open(abs_dir, 'r')
            item = cPickle.load(file_in)
            items = item.split('\n')
            new_con = []
            con = []
            issum = False
            ispred = False
            line_num = 0
            sum_len = len(summary[ord])

            for line in items:
                if 'Summary' in line:
                    continue
                elif line_num < sum_len:
                    text, num = index(line, vocab, vocab_inv, num)
                    if text == summary[ord][line_num]:
                        issum = True
                        line_num = sum_len
                        continue
                elif issum == True and 'Prediction' in line:
                    ispred = True
                    continue
                elif ispred == True and line.strip():
                    text, num = index(line, vocab, vocab_inv, num)
                    if top:
                        con.append(text)
                    else:
                        con += text
            ord += 1
            if top:
                predict.append(con)
            else:
                new_con.append(con)
                predict.append(new_con)
    return num


def data(top, vocab, vocab_inv):
    num = 0
    news_dir = './news_c2/'
    news = []
    print '==== news file ===='
    num = loopDir(news_dir, news, num, vocab, vocab_inv)
    print 'News file number is ', len(news)
    summ_dir = './ground_c2/'
    summary = []
    print '==== Ground Truth file ===='

    num = loopGRD(summ_dir, summary, num, vocab, vocab_inv)
    print 'Summary file number is ', len(summary)
    pred_dir = './best_epoch_predict/'
    predict = []
    num = loopClu(pred_dir, predict, summary, num, vocab, vocab_inv, top)
    # num = loop_beam(pred_dir, predict, num, vocab, vocab_inv)
    print 'Predict file number is ', len(predict)
    return news, summary, predict, vocab, vocab_inv


def to_text(idx, vocab_inv):
    text = ""
    for word in idx:
        if word in vocab_inv:
            text = text + " " + vocab_inv[word]
        else:
            text = text + ' UNK'
    return text.strip()


def oneHot(input, vocab_len):
    enc = np.zeros([1, vocab_len])
    for i in range(vocab_len):
        if i in input:
            enc[0][i] = 1
    return enc


def rouge(pred, reference):
    ROUGE_path = "/home/shirley/rouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl"
    data_path = "/home/shirley/rouge/pythonrouge/RELEASE-1.5.5/data"
    rouge = Pythonrouge(n_gram = 2, ROUGE_SU4 = False, ROUGE_L = False, stemming=True,
                        stopwords = True, word_level = True, length_limit = False, use_cf = False,
                        cf=95, scoring_formula = "average", resampling = True, samples = 1000,
                        favor = True, p = 0.5)
    setting_file = rouge.setting(files = False, summary=pred, reference = reference)
    result = rouge.eval_rouge(setting_file, recall_only=False,
                              ROUGE_path= ROUGE_path, data_path=data_path)
    print result
    return result


def tfidfRep(news, predict, vocab_len):
    news_tfidf = []
    pred_tfidf = []

    for i in range(len(predict)):
        feature = []
        plen = len(predict[i])
        for j in range(plen):
            pone = oneHot(predict[i][j], vocab_len)
            feature.append(pone)
        for k in range(len(news[i])):
            none = oneHot(news[i][k], vocab_len)
            feature.append(none)
        transformer = TfidfTransformer()
        feature = np.asarray(feature)
        feature = np.squeeze(feature)
        tfidf = transformer.fit_transform(feature).toarray()
        pred_temp = tfidf[:plen]
        news_temp = tfidf[plen:]
        pred_tfidf.append(pred_temp)
        news_tfidf.append(news_temp)
    return news_tfidf, pred_tfidf


def cosineSim(news, predict, news_tfidf, pred_tfidf, vocab_inv, summary):  #
    extr_sum = []
    sum_con = []
    """
    extr_dir = './extract_newcen/'
    if not os.path.exists(extr_dir):
        os.makedirs(extr_dir)
    """
    fir_num = 0
    vocab_len = len(vocab_inv)
    for i in range(len(predict)):
        art = news[i]
        pred_sum = predict[i]
        sum_per = []
        pred_enc = []
        for j in range(len(pred_sum)):
            pred_enc = oneHot(pred_sum[j], vocab_len)
            # pred_enc = pred_tfidf[i][j]
            # pred_enc = np.asarray(pred_enc).reshape((1,-1))
        sim_sen = []
        for k in range(len(art)):
            art_enc = oneHot(art[k], vocab_len)
            # art_enc = news_tfidf[i][k]
            # art_enc = np.asarray(art_enc).reshape((1,-1))
            sim_sen.append(cosine_similarity(pred_enc, art_enc)[0,0])
        idx = [m[0] for m in sorted(enumerate(sim_sen), key = lambda x:x[1], reverse=True)]
        sim_sort = [m[1] for m in sorted(enumerate(sim_sen), key = lambda x:x[1], reverse=True)]
        # print 'max_similarity is :\t' , max(sim_sen)
        # print 'top 4 sim after sort is :\t', sim_sort[0:4]
        # print 'the idx is :\t', idx[0:4]
        summ = []
        # out = os.path.join(extr_dir, str(i) + '.txt')
        # ot = open(out, "wb")

        sum_len = len(summary[i])
        """
        for j in range(sum_len):
            if j in idx[0:sum_len]:
                fir_num += 1
                # ot.write("At least one of first" + str(sum_len) + "sentence is extracted!\n")
                break
        
        ot.write("Predicted summary:\n")
        ot.write(to_text(pred_sum[0], vocab_inv) + "\n")
        ot.write("Extracted summary: \n")
        """
        for n in range(sum_len):
            # ot.write(to_text(art[idx[n]], vocab_inv) + "\n")
            summ += art[idx[n]]
        extr_sum.append(summ)
        # ot.write("Real Summary: \n")
        con = []
        for m in range(len(summary[i])):
            # ot.write(to_text(summary[i][m], vocab_inv) + "\n")
            con += summary[i][m]
        sum_per.append(con)
        sum_con.append(sum_per)
        # ot.close()
    print '{} number of file extracted first sentence of news'.format(fir_num)
    return extr_sum, sum_con

def cosineSimTop(news, predict,news_tfidf, pred_tfidf, vocab_inv, summary):
    extr_sum = []
    sum_con = []
    extr_sum_top3 = []
    # extr_dir = './extract_newcen/'
    vocab_len = len(vocab_inv)
    # if not os.path.exists(extr_dir):
    #     os.makedirs(extr_dir)
    fir_num = 0
    for i in range(len(predict)):
        art = news[i]
        pred_sum = predict[i]
        sum_per = []
        pred_enc = []
        top = []
        top_dict = {}
        for j in range(len(pred_sum)):
            pred_enc = oneHot(pred_sum[j], vocab_len)
            # pred_enc = pred_tfidf[i][j]
            # pred_enc = np.asarray(pred_enc).reshape((1,-1))
            sim_sen = []
            for k in range(len(art)):
                art_enc = oneHot(art[k], vocab_len)
                # art_enc = news_tfidf[i][k]
                # art_enc = np.asarray(art_enc).reshape((1,-1))
                sim_sen.append(cosine_similarity(pred_enc, art_enc)[0,0])
            idx = [m[0] for m in sorted(enumerate(sim_sen), key = lambda x:x[1], reverse=True)]
            sim_sort = [m[1] for m in sorted(enumerate(sim_sen), key = lambda x:x[1], reverse=True)]
            for m in range(4):
                if idx[m] not in top_dict:
                    top_dict[idx[m]] = sim_sort[m]
                else:
                    if top_dict[idx[m]] < sim_sort[m]:
                        top_dict[idx[m]] = sim_sort[m]
            for m in range(len(idx)):
                if idx[m] not in top:
                    top.append(idx[m])
                    break
            # print 'max_similarity is :\t' , max(sim_sen)

        # print 'Extracted idx is :\t', top
        # print 'Extracted top 3 idx is:\t', top_dict.keys()
        # print '\n'
        sum_len = len(summary[i])
        """
        out = os.path.join(extr_dir, str(i) + '.txt')
        ot = open(out, "wb")     
        for j in range(sum_len):
            if j in top:
                fir_num += 1
                ot.write("At least one of first" + str(sum_len) + "sentence is extracted!\n")
                break
        ot.write("Predicted summary:\n")
        for h in range(len(pred_sum)):
            ot.write(to_text(pred_sum[h], vocab_inv) + "\n")
        ot.write("Extracted summary: \n")
        """
        top3 = sorted(top_dict, key=top_dict.get, reverse=True)
        t3_summ = []
        summ = []

        for n in range(sum_len):
            # ot.write(to_text(art[top[n]], vocab_inv) + "\n")
            # summ += art[top[n]]
            t3_summ += art[top3[n]]

        if len(pred_sum) < 4:
            pd_len = len(pred_sum)
        else:
            pd_len = 4
        for n in range(pd_len):
            summ += art[top[n]]
        extr_sum.append(summ)
        extr_sum_top3.append(t3_summ)
        # ot.write("Real Summary: \n")
        con = []
        for m in range(len(summary[i])):
            # ot.write(to_text(summary[i][m], vocab_inv) + "\n")
            con += summary[i][m]
        sum_per.append(con)
        sum_con.append(sum_per)
        # ot.close()
    print '{} number of file extracted first sentence of news'.format(fir_num)
    return extr_sum, sum_con, extr_sum_top3


# -- Main --
if __name__ == '__main__':
    top = True # False #
    vocab = {}
    vocab_inv = {}
    news, summary, predict, vocab, vocab_inv = data(top, vocab, vocab_inv)
    vocab_len = len(vocab)
    print 'vocab_len: \t', vocab_len
    # news_tfidf, pred_tfidf = tfidfRep(news, predict, vocab_len)
    news_tfidf, pred_tfidf = [], []
    extr_sum, reference, extr_sum_top3 = cosineSimTop(news, predict, news_tfidf, pred_tfidf, vocab_inv, summary)
    print 'Extract 3 rouge:'
    rouge(extr_sum_top3, reference)
    # extr_sum, reference = cosineSim(news, predict,news_tfidf, pred_tfidf, vocab_inv, summary)
    # rouge(extr_sum, reference)
