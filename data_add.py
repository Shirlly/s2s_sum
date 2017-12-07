import os
import cPickle
import numpy as np


def index(words, vocab, vocab_inv, num):
    text = []
    for word in words:
        if word not in vocab and word.strip() is not 'RT' and word.strip() is not 'rt':
            vocab[word] = num
            vocab_inv[num] = word
            num = num + 1
        text.append(vocab[word])
    return text, num

def index_infer(words, vocab, vocab_inv, num):
    text = []
    for word in words:
        if word not in vocab and word.strip() is not 'RT' and word.strip() is not 'rt':
            print '+++ unknown words is ', word
            text.append('UNK')
        else:
            text.append(vocab[word])
    return text, num


def data(dir, file, vocab, vocab_inv, max_twee_len, num, twe_num_pnew, news_num):
    eml = 0
    emn = 0
    summ_temp = []
    news_temp = []
    news_text = []
    twee_temp = []
    sum_org = []
    news_org = []
    tweet_org = []
    title = []
    first = []
    tweet_dict = {}
    for line in file:
        if not line.strip() and eml == 0:
            eml = eml + 1
        elif not line.strip() and eml == 1:
            eml = 0
            emn = emn + 1
        elif line.strip() and eml == 1:
           eml = 0

        if line.strip() and emn == 0:
            sum_org.append(line.strip())
            words = line.strip().split()
            text, num = index(words, vocab, vocab_inv, num)
            # summ_temp = summ_temp + text
            summ_temp.append(text)
        elif line.strip() and emn == 1:
            news_org.append(line.strip())
            words = line.strip().split()
            text, num = index(words, vocab, vocab_inv, num)
            news_temp = news_temp + text
            news_text.append(text)
        elif line.strip() and emn == 2:
            words = line.strip().split()
            con = " ".join(words)
            if not con in tweet_dict:
                tweet_org.append(line.strip())
                tweet_dict[con] = 1
                text, num = index(words, vocab, vocab_inv, num)
                twee_temp.append(text)

    """
    sum_ele_max_len = len(max(summ_temp, key=len))
    if max_sum_len < sum_ele_max_len:
        max_sum_len = sum_ele_max_len

    rt_ele_max_len = len(max(twee_temp, key=len))
    if max_rt_len < rt_ele_max_len:
        max_rt_len = rt_ele_max_len
        max_twee = max(twee_temp, key=len)
        idx = twee_temp.index(max_twee)
        max_twee.append(dir)
        max_twee.append(idx)
    
    for k in range(5):
        for i in range(2):
            twee_temp.append(news_text[i])
    """
    twee_len = len(twee_temp)
    twe_num_pnew.append(twee_len)
    news_len = len(news_text)
    news_num.append(news_len)
    if not twee_len:
        print 'File ', dir
        print "#### No tweets ####"
    else:
        news_twee_temp = [news_temp]*twee_len
        title = [news_text[0]] * twee_len
        first = [news_text[1]] * twee_len
        news_twee_len = len(news_twee_temp)
        if news_twee_len != twee_len:
            print "***** Error matching! *****"
        twee_ele_max_len = len(max(twee_temp, key=len))
        if max_twee_len < twee_ele_max_len:
            max_twee_len = twee_ele_max_len
    return summ_temp, news_temp, news_twee_temp, twee_temp, max_twee_len, num, news_text, \
           title, first, sum_org, news_org, tweet_org


def loop_dir():
    file_dir = './clean2/'
    sum_org = []
    news_org = []
    tweet_org = []
    summary = []
    news = []
    news_con = []
    news_twee = []
    tweets = []
    title = []
    first = []
    twe_num_pnew = []
    news_num = []
    vocab = {}
    vocab_inv = {}
    count = 0
    max_twee_len = 0
    # max_sum_len = 0
    # max_rt_len = 0
    # max_twee = []
    num = 2
    for filename in sorted(os.listdir(file_dir), key = int):
        print 'filename ', filename
        abs_dir = os.path.join(file_dir, filename)
        if os.path.isfile(abs_dir):
            file_in = open(abs_dir, 'r')
            summ_temp, news_temp, news_twee_temp, twee_temp, max_twee_len, num, news_text, title_temp, \
                first_temp, sum_temp, ne_temp, tweet_temp = \
                data(abs_dir, file_in, vocab, vocab_inv, max_twee_len, num, twe_num_pnew, news_num)
            sum_org.append(sum_temp)
            news_org.append(ne_temp)
            tweet_org.append(tweet_temp)
            summary.append(summ_temp)
            news.append(news_temp)
            news_con.append(news_text)
            news_twee = news_twee + news_twee_temp
            tweets = tweets + twee_temp
            title = title + title_temp
            first = first + first_temp
            count = count + 1
            print '==== Data reading from ', filename, ' is done!'
    print 'max_tweet_length ', max_twee_len
    print 'size of summary is {} and {} '.format(len(summary), len(summary[0]))
    print 'size of news is {} and {}'.format(len(news), len(news[0]))
    print 'size of news_twee is {} and {}'.format(len(news_twee), len(news_twee[0]))
    print 'size of tweet is {} and {}'.format(len(tweets), len(tweets[0]))

    if 'RT' in vocab:
        print '==== remove RT ===='
        idx = vocab['RT']
        del vocab['RT']
        del vocab_inv[idx]
    return summary, news, news_twee, tweets, vocab, vocab_inv, max_twee_len, news_con, title, first, \
           twe_num_pnew, news_num, sum_org, news_org, tweet_org


def infer_data(item, vocab, vocab_inv, num, summi):
    sum_ind = False
    clu_infer = []
    clu_infer_con = []
    clu_temp = []
    con_temp = []
    sum_num = 0
    sum_len = len(summi)
    items = item.split('\n')
    for line in items:
        if 'Summary' in line:
            continue
        elif sum_num<sum_len:
            words = line.strip().split()
            text, num = index_infer(words, vocab, vocab_inv, num)
            if text == summi[sum_num]:
                sum_ind = True
                sum_num = sum_len
            else:
                '=== Document matching wrong! ==='
        elif sum_num == sum_len and 'Cluster' in line:
            sum_ind = False
            if len(clu_temp) > 0:
                clu_infer.append(clu_temp)
                clu_infer_con.append(con_temp)
            clu_temp = []
            con_temp = []
        elif sum_ind == False:
            words = line.strip().split()
            # text, num = index(words, vocab, vocab_inv, num)
            text, num = index_infer(words, vocab, vocab_inv, num)
            clu_temp += text
            con_temp.append(text)

    if len(clu_temp) > 0:
        clu_infer.append(clu_temp)
        clu_infer_con.append(con_temp)

    if len(clu_infer) != sum_len:
        import pdb;pdb.set_trace()
        print '+++++ Matching wrong! ++++'
    return clu_infer, clu_infer_con


def load_infer_data(file_dir, vocab, vocab_inv, summary):
    infer = []
    infer_con = []
    num = 0
    snum = 0
    for filename in sorted(os.listdir(file_dir)):  # , key=int
        print 'filename ', filename
        summi = summary[snum]
        abs_dir = os.path.join(file_dir, filename)
        if os.path.isfile(abs_dir):
            file_in = open(abs_dir, 'r')
            items = cPickle.load(file_in)
            clu_infer, clu_infer_con = infer_data(items, vocab, vocab_inv, num, summi)
            infer += clu_infer
            infer_con += clu_infer_con
        snum += 1
    if num == 0:
        print '=== Vocabulary is correct! ==='
    return infer, infer_con
