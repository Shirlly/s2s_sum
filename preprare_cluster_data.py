import os
import cPickle
import hdbscan
import numpy as np
import pandas as pd
import tensorflow as tf
from helpers import batch
from unsupervisedS2Sadv1 import unsuS2S_adv
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
from cluster_process import cluster_demo, grd_news


def to_text(idx, vocab_inv):
    text = ""
    for word in idx:
        if word in vocab_inv:
            text = text + " " + vocab_inv[word]
        else:
            text = text + ' UNK'
    return text.strip()


def writeText(text_list, idx, vocab_inv):
    text = ''
    for i in range(len(idx)):
        text += to_text(text_list[idx[i]], vocab_inv) + "\n"
    return text


def tweeClu(estimator, tweet_rep, ele_idx):
    ele_dis = []
    twee_lab = []
    for i in range(len(ele_idx)):
        enc = np.reshape(tweet_rep[ele_idx[i]], [1, -1])
        lab = estimator.predict(enc)
        dis = estimator.transform(enc)
        twee_lab.append(lab[0])
        ele_dis.append(dis[0][lab[0]])
    return twee_lab, ele_dis


def tweet_news_msim(tweet_rep, new_rep, twee_lab, news_lab):
    clu_add_tweet = {}
    for i in range(len(tweet_rep)):
        sim = {}
        for j in range(len(new_rep)):
            n_rep = np.reshape(np.asarray(new_rep[j]), [1, -1])
            t_rep = np.reshape(np.asarray(tweet_rep[i]), [1, -1])
            sim[j] = cosine_similarity(n_rep, t_rep)[0, 0]
        sortn = sorted(sim, key=sim.get, reverse=True)
        news_clu = [news_lab[m] for m in [sortn[0]]]
        exist = []
        for k in news_clu:
            if isinstance(k, list) and k not in exist:
                for ele in k:
                    if twee_lab[i] != ele and ele in clu_add_tweet:
                        at = clu_add_tweet[ele]
                        at.append(i)
                        clu_add_tweet[ele] = at
                    elif twee_lab[i] != ele and ele not in clu_add_tweet:
                        clu_add_tweet[ele] = [i]
            elif k != -1 and k not in exist:
                if twee_lab[i] != k and k in clu_add_tweet:
                    at = clu_add_tweet[k]
                    at.append(i)
                    clu_add_tweet[k] = at
                elif twee_lab[i] != k and k not in clu_add_tweet:
                    clu_add_tweet[k] = [i]
            exist.append(k)
    return clu_add_tweet


def concateAspIdx(con, idx, concate):
    cct = []
    for i in range(len(idx)):
        if concate:
            cct += con[idx[i]]
        else:
            cct.append(con[idx[i]])
    return cct


def tweet_back(clu, news_out, twee_out, new_rep, tweet_rep, center):
    twee_out_fin = []
    cop_news = []
    for ele in twee_out:
        less_num = 0
        for ne in news_out:
            n_rep = np.reshape(np.asarray(new_rep[ne]), [1, -1])
            t_rep = np.reshape(np.asarray(tweet_rep[ele]), [1, -1])
            c_rep = np.reshape(np.asarray(center[clu]), [1, -1])
            news_twee_sim = cosine_similarity(n_rep, t_rep)[0, 0]
            core_twee_sim = cosine_similarity(t_rep, c_rep)[0, 0]
            """
            mem = [new_rep[ne]] + [tweet_rep[ele]]
            news_twee_sim = pairwise_distances(X=mem, metric='euclidean')[0, 1]   # canberra
            mem = [new_rep[ne]] + [center[clu].tolist()]
            core_twee_sim = pairwise_distances(X=mem, metric='euclidean')[0, 1]   # canberra
            # core_twee_sim = cosine_similarity(new_rep[ne], estimator.cluster_centers_[clu])[0, 0]
            """
            if news_twee_sim < core_twee_sim:
                less_num += 1
            else:
                cop_news.append(ne)
        if less_num == len(news_out):
            twee_out_fin.append(ele)
        """
        else:
            print '#### Tweet {} of doc {} get back '.format(ele, i)
            print '#### Similar news is ',cop_news
        """
    return twee_out_fin


def assignTweet(cluster, news_rep, twee_rep, tweets, news_con, news_num, summary, vocab_inv):
    encoder_input = []
    decoder_target = []
    decoder_infer_tar = []
    decoder_infer_pin = []
    data_info = []
    clu_sen_len = []

    clu_dir = './km_avgT/'
    if not os.path.exists(clu_dir):
        os.makedirs(clu_dir)
    cluster_mem_num = []
    for i in range(len(cluster)):
        tweet_rep = twee_rep[i]
        tweet = tweets[i]
        ele_idx = range(0, len(tweet))

        new_rep = news_rep[i]

        clu_file = open(clu_dir + str(i) + '.txt', 'wb')
        text = 'Summary:\n'
        sidx = range(0, len(summary[i]))
        text += writeText(summary[i], sidx, vocab_inv)

        label = cluster[i].labels_
        twee_lab, ele_dis = tweeClu(cluster[i], tweet_rep, ele_idx)
        clu_add_tweet = tweet_news_msim(tweet_rep, new_rep, twee_lab, label)
        for j in range(len(set(label))):
            news_idx = [m for m, x in enumerate(label) if x == j]
            twee_idx = [m for m, x in enumerate(twee_lab) if x == j]
            news_idx = sorted(news_idx)

            if j in clu_add_tweet:
                twee_idx += clu_add_tweet[j]

            if len(twee_idx) > 0:
                twee_idx_temp = [m + news_num[i] for m in twee_idx]
            else:
                twee_idx_temp = []
            nt_idx = news_idx + twee_idx_temp
            nt_rep = []
            nt_rep += new_rep
            nt_rep += tweet_rep

            text += 'Cluster ' + str(j) + ':\tNews: ' + str(len(news_idx)) + '\n'
            text += writeText(news_con[i], news_idx, vocab_inv)
            text += writeText(news_con[i], [0, 1], vocab_inv)

            if len(nt_idx) > 1:
                estimator = hdbscan.HDBSCAN(min_cluster_size=2)  # , algorithm='prims_kdtree'
                result = estimator.fit(nt_rep)
                threshold = pd.Series(result.outlier_scores_).quantile(0.8)
                outliers = np.where(result.outlier_scores_ > threshold)[0]

                news_out = [m for m in outliers if m in news_idx]
                twee_out = [m for m in outliers if m in twee_idx_temp]
                true_twee_out = [m - news_num[i] for m in twee_out]
                center = cluster[i].cluster_centers_
                twee_out_fin = tweet_back(j, news_out, true_twee_out, new_rep, tweet_rep, center)
                twee_idx_fin = [m for m in twee_idx if m not in twee_out_fin]
            else:
                print 'No Element in cluster {} of doc {}'.format(j, i)
                twee_idx_fin = [m - news_num[i] for m in twee_idx_temp]

            temp_n = []
            nt_con = []
            temp_t = []

            if len(news_idx) > 0:
                temp_n = concateAspIdx(news_con[i], news_idx, concate=True)
                nt_con = concateAspIdx(news_con[i], news_idx, concate=False)

            temp_n += concateAspIdx(news_con[i], [0, 1], concate=True)
            nt_con += concateAspIdx(news_con[i], [0, 1], concate=False)

            temp_t = concateAspIdx(news_con[i], [0, 1], concate=False)
            if len(twee_idx_fin) > 0:
                temp_t += concateAspIdx(tweet, twee_idx_fin, concate=False)

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
        cPickle.dump(text, clu_file)
    return encoder_input, decoder_target, decoder_infer_tar, decoder_infer_pin, \
           data_info, clu_sen_len, cluster_mem_num


def conv3to2(news_con):
    news2 = []
    news_num = []
    for i in range(len(news_con)):
        nlen = len(news_con[i])
        news_num.append(nlen)
        for j in range(nlen):
            news2.append(news_con[i][j])
    return news2, news_num


def conv2to3(tweets, twee_num, news_con):
    twee3 = []
    for i in range(len(news_con)):
        snum = sum(twee_num[:i])
        enum = snum + twee_num[i]
        twee3.append(tweets[snum:enum])
    return twee3


def next_feed_inference(news, encoder_input, en_in_len):
    _encoder, _encoder_len, _ = batch(news)
    return {encoder_input: _encoder, en_in_len: _encoder_len}


def cluster_data(args, vocab_size, pre_embed, news_con, tweets, twe_num):
    news2, news_num = conv3to2(news_con)

    s2s = unsuS2S_adv(vocab_size, args.embed_dim, args.batch_size, pre_embed, attention=True)
    # s2s = Seq2Seq(vocab_size, args.embed_dim, pre_embed, 128, 256, args.cell, args.beam_size, 30)

    with s2s.graph.as_default():
        encoder_input, en_in_len, encoder_output, encoder_final_state = s2s.encoder()
        saver = tf.train.Saver()
        if args.cluster_rep_restore == True:
            print 'Cluster restore True'
            try:
                saver.restore(s2s.sess, './log2_gru_embed21/pre_check_point-49')
            except:
                import pdb; pdb.set_trace()
        else:
            print 'Cluster restore False'
            tf.set_random_seed(1)
            init = tf.global_variables_initializer()
            s2s.sess.run(init)

        feed_dict = next_feed_inference(news2, encoder_input, en_in_len)
        news_state = s2s.sess.run(encoder_final_state, feed_dict)
        sen_rep = news_state  # news_state.h  #
        news_rep = sen_rep.tolist()
        news3 = conv2to3(news_rep, news_num, news_con)

        feed_dict = next_feed_inference(tweets, encoder_input, en_in_len)
        twee_state = s2s.sess.run(encoder_final_state, feed_dict)
        sen_rep = twee_state  # twee_state.h  #
        twee_rep = sen_rep.tolist()
        twee3 = conv2to3(twee_rep, twe_num, news_con)

        return news3, twee3, news_num



def train_data(args, vocab_size, pre_embed, news_con, tweets, twe_num, summary, vocab_inv, news_grd_sim_rank):
    news3, twee3, news_num = cluster_data(args, vocab_size, pre_embed, news_con, tweets, twe_num)
    tweet3 = conv2to3(tweets, twe_num, news_con)
    if len(news_num) == len(twe_num) == len(summary):
        cluster = cluster_demo(news3, news_num, summary)
        print 'Clustering Done'

    encoder_in, decoder_tar, decoder_infer_tar, decoder_infer_pin, data_info, clu_sen_len, \
        cluster_mem_num = assignTweet(cluster, news3, twee3, tweet3, news_con, news_num,
                                  summary, vocab_inv)

    train_encoder_in, train_decoder_tar, valid_encoder_in, valid_decoder_tar = encoder_in, decoder_tar, [], []

    train_dir = args.log_root + '_' + args.opt + '_' + str(args.opt_clip) + '_plain'
    if args.all == True:
        train_dir += '_all'
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

