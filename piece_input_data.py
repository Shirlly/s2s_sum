import os
import math
import cPickle
import hdbscan
import pandas as pd
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from lex_rank import LexRankSummarizer
from preprare_cluster_data import cluster_data, tweet_back, concateAspIdx, conv2to3
from sklearn.metrics.pairwise import cosine_similarity
from lexrank import tweet_vote


def sum_news(news_org, news_grd_sim_rank, grd_summary, news_con, summary_con, news3, pre_p=False):
    file_dir = './news_text/'
    if pre_p == True:
        p_vector = tweet_vote(news_con)
    tlt_per = []
    tlt_cluster = []
    de_news = []
    for filename in sorted(os.listdir(file_dir), key=int):
        # print 'filename ', filename
        i = int(filename)
        abs_dir = os.path.join(file_dir, filename)
        avg_p, matrix = [], []
        if pre_p == True:
            avg_p = p_vector[i]
        """
        avg_p = np.divide(np.asarray(p_vector[i]), sum(p_vector[i]))
        news_rep = news3[i]
        matrix = cosine_similarity(news_rep)
        matrix[matrix < 0.3] = 0
        """
        parser = PlaintextParser.from_file(abs_dir, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary, score = summarizer(parser.document, len(grd_summary[i]), avg_p, matrix)
        new_score = {str(key): score[key] for key in score}

        lower_news = []
        for news in news_org[i]:
            lower_news.append(news.lower())

        if len(news_con[i]) < (7*len(grd_summary[i])):
            doc_len = math.floor((len(news_con[i]) * 1.0)/len(grd_summary[i]))
            cluster_ele, de_news_ele = avg_split(int(doc_len), len(grd_summary[i]), new_score, lower_news)
            per = 1
        else:
            per, cluster_ele, de_news_ele = news_range(lower_news, news_grd_sim_rank[i],
                                                       news_con[i], i, new_score, grd_summary[i])
        tlt_cluster.append(cluster_ele)
        tlt_per.append(per)
        de_news.append(de_news_ele)

    print "Total percentage of news is ", sum(tlt_per)/len(tlt_per)
    return tlt_cluster, de_news


def avg_split(doc_len, clu_num, new_score, lower_news):
    sort_score = sorted(new_score, key=new_score.get, reverse=True)
    rk = []
    for n in lower_news:
        rk.append(sort_score.index(n))
    cluster_ele = {}
    de_news_ele = {}
    for i in range(clu_num):
        cluster_ele[i] = range(i*doc_len, (i+1)*doc_len)
        min_rk_id = np.argmin(np.asarray(rk[i*doc_len:(i+1)*doc_len]))
        de_news_ele[i] = [min_rk_id]
    return cluster_ele, de_news_ele


def news_range(lower_news, news_grd_sim_rank, news_con, i, score, summary):
    rk_li = np.asarray(news_grd_sim_rank)
    top_idx = []
    for k in range(rk_li.shape[1]):
        idx = np.where(rk_li[:, k] < 3)[0].tolist()
        top_idx += idx
    cluster = {}
    clu_ele = []
    inter = {}
    num = 0
    sort_score = sorted(score, key=score.get, reverse=True)
    de_news_ele = {}
    for sentence in sort_score:
        if len(cluster) < len(summary):
            sentence = str(sentence)
            news_id = lower_news.index(sentence)
            id_range = []
            if news_id not in clu_ele:
                if news_id - 3 <= 0:
                    id_range = range(0, 7)
                elif (len(news_con) - news_id) <= 3:
                    doc_len = len(news_con)
                    id_range = range(doc_len - 7, doc_len - 1)
                else:
                    id_range = range(news_id - 3, news_id + 4)
                add = True

                for clu in cluster:
                    inter_len = len(list(set(id_range).intersection(set(cluster[clu]))))
                    if inter_len > 0:
                        inter[news_id] = id_range
                        clu_ele.append(news_id)
                        add = False
                if add == True:
                    cluster[num] = id_range
                    de_news_ele[num] = [news_id]
                    clu_ele += id_range
                    num += 1

    while len(cluster) < len(summary):
        cluster, de_news_ele, inter = split_cluster(cluster, de_news_ele, inter)

    for nid in inter:
        inter_len = {}
        for clu in cluster:
            in_len = len(list(set(inter[nid]).intersection(set(cluster[clu]))))
            inter_len[clu] = in_len
        sort_inter = sorted(inter_len, key=inter_len.get, reverse=True)
        clu = sort_inter[0]
        cluster[clu].append(nid)
        de_news_ele[clu].append(nid)

    min_ele = {}
    core_ele = [0, 1]
    for e in core_ele:
        if e not in clu_ele:
            for clu in cluster:
                min_ele[clu] = min(cluster[clu])
            sort_min_ele = sorted(min_ele, key=min_ele.get)
            mele = sort_min_ele[0]
            cluster[mele].append(e)
            de_news_ele[mele].append(e)
    clu_ele += core_ele
    common = list(set(top_idx).intersection(set(clu_ele)))
    per = (len(common) * 1.0) / len(set(top_idx))
    """
    print "Percentage {} has been included in selected parts".format(per)
    print "Top news id ", top_idx
    print "Document length ", len(score)
    print "Cluster ", cluster """

    return per, cluster, de_news_ele


def split_cluster(cluster, de_news_ele, inter):
    min_inter_len = {}
    for ele in inter:
        inter_len = {}
        for clu in cluster:
            in_len = len(list(set(inter[ele]).intersection(set(cluster[clu]))))
            inter_len[clu] = in_len
        sort_inter = sorted(inter_len, key=inter_len.get)
        try:
            min_ele = sort_inter[0]
        except:
            import pdb; pdb.set_trace()
        min_inter_len[ele] = inter_len[min_ele]
    sort_min = sorted(min_inter_len, key=min_inter_len.get)
    try:
        min_add = sort_min[0]
    except:
        import pdb; pdb.set_trace()
    clu_num = len(cluster)
    cluster[clu_num] = inter[min_add]
    de_news_ele[clu_num] = [min_add]
    del inter[min_add]
    return cluster, de_news_ele, inter


def tweet_news_msim(tweet_rep, new_rep, news_lab):
    clu_add_tweet = {}
    for i in range(len(tweet_rep)):
        sim = {}
        nlen = len(new_rep)
        for j in range(nlen):
            n_rep = np.reshape(np.asarray(new_rep[j]), [1, -1])
            t_rep = np.reshape(np.asarray(tweet_rep[i]), [1, -1])
            sim[j] = cosine_similarity(n_rep, t_rep)[0, 0]
        sortn = sorted(sim, key=sim.get, reverse=True)
        news_clu = [news_lab[m] for m in sortn[:2]]
        for k in set(news_clu):
            if k != -1 and k in clu_add_tweet:
                at = clu_add_tweet[k]
                at.append(i)
                clu_add_tweet[k] = at
            elif k not in clu_add_tweet:
                clu_add_tweet[k] = [i]
    return clu_add_tweet


def assignTweet(tlt_cluster, tlt_label, news_rep, twee_rep, tweets, news_con, summary, de_news):
    encoder_input = []
    decoder_target = []
    decoder_infer_tar = []
    decoder_infer_pin = []
    data_info = []
    clu_sen_len = []

    cluster_mem_num = []
    for i in range(len(tlt_cluster)):
        tweet_rep = twee_rep[i]
        tweet = tweets[i]

        new_rep = news_rep[i]

        label = tlt_label[i]
        clu_tweet = tweet_news_msim(tweet_rep, new_rep, label)
        for j in range(len(summary[i])):
            news_idx = tlt_cluster[j]
            twee_idx = []
            if j in clu_tweet:
                twee_idx = clu_tweet[j]
            news_idx = sorted(news_idx)

            temp_n = []
            nt_con = []
            temp_t = []

            if len(news_idx) > 0:
                temp_n = concateAspIdx(news_con[i], news_idx, concate=True)
                nt_con = concateAspIdx(news_con[i], news_idx, concate=False)

            # temp_n += concateAspIdx(news_con[i], [0, 1], concate=True)
            # nt_con += concateAspIdx(news_con[i], [0, 1], concate=False)
            try:
                temp_t = concateAspIdx(news_con[i], de_news[i][j], concate=False)
                for k in range(4):
                    temp_t += concateAspIdx(news_con[i], de_news[i][j], concate=False)
                if 0 not in de_news[i][j]:
                    for k in range(5):
                        temp_t += concateAspIdx(news_con[i], [0], concate=False)
            except:
                import pdb; pdb.set_trace()

            if len(twee_idx) > 0:
                temp_t += concateAspIdx(tweet, twee_idx, concate=False)

            sen_len = [len(news_con[i][j]) for j in news_idx]

            twee_len = len(temp_t)
            cluster_mem_num.append(twee_len)
            encoder = [temp_n] * twee_len
            info = [{str(i) + ',' + str(j): news_idx}] * twee_len
            temp_sen_len = [sen_len] * twee_len
            encoder_input += encoder
            decoder_target += temp_t
            data_info += info
            clu_sen_len += temp_sen_len
            decoder_infer_tar.append(temp_n)
            decoder_infer_pin.append(nt_con)
    return encoder_input, decoder_target, decoder_infer_tar, decoder_infer_pin, \
           data_info, clu_sen_len, cluster_mem_num


def labels(tlt_cluster, news_con):
    tlt_label = []
    for i in range(len(tlt_cluster)):
        clu_label = np.full((len(news_con[i])), -1, dtype=int)
        for clu in tlt_cluster[i]:
            for idx in tlt_cluster[i][clu]:
                clu_label[idx] = clu
        tlt_label.append(clu_label.tolist())
    return tlt_label


def piece_data(args, vocab_size, pre_embed, summary, news_con, tweets, twe_num, news_org, sum_org,
               news_grd_sim_rank):
    news3, twee3, news_num = cluster_data(args, vocab_size, pre_embed, news_con, tweets, twe_num)
    tlt_cluster, de_news = sum_news(news_org, news_grd_sim_rank, sum_org, news_con, summary, news3,
                                    pre_p=args.prep)
    tlt_label = labels(tlt_cluster, news_con)

    tweet3 = conv2to3(tweets, twe_num, news_con)
    encoder_in, decoder_tar, decoder_infer_tar, decoder_infer_pin, data_info, clu_sen_len, \
        cluster_mem_num = assignTweet(tlt_cluster, tlt_label, news3, twee3, tweet3, news_con, summary, de_news)

    train_encoder_in, train_decoder_tar, valid_encoder_in, valid_decoder_tar = encoder_in, decoder_tar, [], []

    print "Encoder in and tar length is ", len(train_decoder_tar)

    train_dir = args.log_root + '_' + args.opt + '_' + '_piece'

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    print 'Saving data'
    data = {'train_encoder_in': train_encoder_in, 'train_decoder_tar': train_decoder_tar,
            'decoder_infer_tar': decoder_infer_tar, 'decoder_infer_pin': decoder_infer_pin,
            'data_info': data_info, 'clu_sen_len': clu_sen_len}
    cPickle.dump(data, open(train_dir + '/all_data', 'wb'))
    print 'Saving data done'

    return train_encoder_in, train_decoder_tar, decoder_infer_tar, decoder_infer_pin, \
           data_info, clu_sen_len, train_dir
