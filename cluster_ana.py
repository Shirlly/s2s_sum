import os
import hdbscan
import cPickle
import pandas as pd
import numpy as np
import math
from data_add import loop_dir
from difflib import SequenceMatcher as sm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
from sklearn.preprocessing import normalize
# from extractSummary import rouge
from kmeans_staff1 import oneHot, tweeClu, writeText, concateAspIdx, topCenterTxt
from kmeans_staff import prepareData, news_twee_tfidf
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, \
    AffinityPropagation, DBSCAN
from sklearn.feature_extraction.text import TfidfTransformer
from unsupervisedS2Sadv1 import unsuS2S_adv
# from main_clu1 import next_feed_inference, conv3to2
from cluster_quality import grd_news, top_quality, tweet_cluster, tweet_org_cluster
import tensorgraph as tg
import tensorflow as tf


def check_core(labels, news_idx):
    sort_idx = sorted(news_idx)
    core_label = [x for i, x in enumerate(labels) if i in sort_idx]
    diff_label_num = len(set(core_label))
    # print '++ core news {} labels are {}'.format(sort_idx, core_label)
    return diff_label_num


def cluster_choice(tnews, clu, clu_num):
    if clu == 'dbscan':
        estimator = DBSCAN(eps=0.001, min_samples=2)
    elif clu == 'agg':
        estimator = AgglomerativeClustering(n_clusters=clu_num)
    elif clu == 'kmeans':
        estimator = KMeans(init='k-means++', n_clusters=clu_num, random_state=16)
    elif clu == 'spectral':
        estimator = SpectralClustering(n_clusters=clu_num, random_state=0,
                                       affinity='precomputed',   #nearest_neighbors
                                       # n_neighbors=7,
                                       assign_labels='discretize')
    elif clu == 'affinity':
        estimator = AffinityPropagation()
    result = estimator.fit(tnews)
    return result


def reassign_clu(result, clu_num):
    sample = result.cluster_centers_
    estimator = AgglomerativeClustering(n_clusters=clu_num)
    out = estimator.fit(sample)
    sample_label = out.labels_
    org_label = result.labels_
    new_label = []
    for i in range(len(org_label)):
        sample_num = org_label[i]
        new_label.append(sample_label[sample_num])
    return new_label


def doc_tfidf(doc):
    vocab = {}
    vocab_inv = {}
    doc_rep = []
    num = 0
    for sen in doc:
        sen_rep = []
        for word in sen:
            if word in vocab:
                sen_rep.append(vocab[word])
            else:
                vocab[word] = num
                sen_rep.append(num)
                vocab_inv[num] = word
                num += 1
        doc_rep.append(sen_rep)
    feature = []
    vocab_len = len(vocab)
    for i in range(len(doc_rep)):
        enc = oneHot(doc_rep[i], vocab_len)
        feature.append(enc)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(feature).toarray()
    return tfidf


def cluster_size_check(result, clu, clu_num, tnews, news_idx):
    small_ele = []
    large_ele = []
    re_clu = []
    for i in range(clu_num):
        temp_ele = [m for m, x in enumerate(result.labels_) if x == i]
        temp_size = len(temp_ele)
        if temp_size < 5:
            small_ele += temp_ele
            print '===== Cluster in doc {} less than 5 element, size is{}'.format(clu, temp_size)
        else:
            large_ele += temp_ele
    if len(small_ele) > 0:
        for j in large_ele:
            re_clu.append(tnews[j])
        for m in news_idx:
            if m in small_ele:
                print '********Core sentence in small cluster'
    return re_clu, small_ele, large_ele

def org_sim(news_org):
    sim_mat = []
    for i in range(len(news_org)):
        row_sim = []
        for j in range(len(news_org)):
            row_sim.append(sm(None, news_org[i], news_org[j]).ratio())
        sim_mat.append(row_sim)
    return sim_mat


def cluster_demo(news_rep, news_num, summary, news_idx):
    cluster = []
    label_num = 0
    tlt_num = 0
    for i in range(len(news_num)):
        clu_num = len(summary[i])
        # Document specific tfidf embedding
        # tnews = doc_tfidf(news_con[i])
        # pre-trained embedding

        tnews = news_rep[i]
        """
        ntwee = sum(twee_num[:i])
        tend = ntwee + twee_num[i]
        twee = twee_rep[ntwee:tend]
        feature = tnews + twee
        
        sim_mat = org_sim(news_org[i])
        """
        clu = 'kmeans'
        result = cluster_choice(tnews, clu, clu_num)
        cluster.append(result)
        # check_core_top(news_num[i],  result, tnews, news_idx[i])
        # result_new = result.labels_
        # res_clu_num = len(set(result.labels_))
        # if res_clu_num > clu_num:
        #     result_new = reassign_clu(result, clu_num)

        # re_clu, small_ele, large_ele = cluster_size_check\
        #     (result, i, clu_num, tnews, news_idx[i])
        # if len(re_clu) > 0:
        #     print 'Cluster in doc {} is less than 5'.format( i)
        #     re_cluster()

        if len(news_idx) > 0:
            label_num += check_core(result.labels_, news_idx[i])
            tlt_num += len(news_idx[i])
            # print 'Noise percentage is:\t', (len(noise)*1.0)/len(tnews)
            # print 'number of clusters of doc {} is {}'.format(i, len(set(result.labels_)))
            # print 'Percentage of different labels is: ', (label_num * 1.0) / tlt_num
    return cluster


def tweet_back(clu, news_out, twee_out, new_rep, tweet_rep, center):
    twee_out_fin = []
    cop_news = []
    for ele in twee_out:
        less_num = 0
        for ne in news_out:
            news_twee_sim = cosine_similarity(new_rep[ne], tweet_rep[ele])[0, 0]
            core_twee_sim = cosine_similarity(tweet_rep[ele], center[clu])[0, 0]
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


def check_core_news(news_idx, twee_idx, new_rep, tweet_rep, top_news_grd_ele, rank_ele, aet3, aet_tlt, et_base):
    near_news = {}
    near_news_sim = {}
    all = 0
    for ti in twee_idx:
        nt_sim = []
        for ni in news_idx:
            news_twee_sim = cosine_similarity(new_rep[ni], tweet_rep[ti])[0, 0]
            nt_sim.append(news_twee_sim)

        nt_news_id = [m[0] for m in sorted(enumerate(nt_sim), key=lambda x: x[1], reverse=True)]
        nt_news_sim = [m[1] for m in sorted(enumerate(nt_sim), key=lambda x: x[1], reverse=True)]
        nt_num = 0

        id = 0
        for ni in nt_news_id[:5]:
            if news_idx[ni] in near_news_sim:
                score = near_news_sim[news_idx[ni]]
                score += nt_news_sim[id]
                near_news_sim[news_idx[ni]] = score
            else:
                near_news_sim[news_idx[ni]] = nt_news_sim[id]
            id += 1

        for ni in nt_news_id[:5]:
            all += 1
            if nt_news_sim[nt_num] > 0.7 and news_idx[ni] in near_news:  #
                num = near_news[news_idx[ni]]
                num += 1
                near_news[news_idx[ni]] = num
            elif nt_news_sim[nt_num] > 0.7 and news_idx[ni] not in near_news:  #
                near_news[news_idx[ni]] = 1
            nt_num += 1
    """
    for ele in near_news:
        avg = near_news_sim[ele]/near_news[ele]
        near_news_sim[ele] = avg
    """
    arr = []
    arr_ele = []
    try:
        for ele in near_news_sim:
            arr.append(near_news_sim[ele])
            arr_ele.append(ele)
        arr_norm = normalize(arr)[0]
        for j in range(len(arr_norm)):
            near_news_sim[arr_ele[j]] = arr_norm[j]
    except:
        print 'No tweets in the cluster'

    sort_nidx = sorted(near_news_sim, key=near_news.get, reverse=True)
    et_tlt = 0
    for ele in news_idx:
        if ele in top_news_grd_ele:
            et_tlt += 1

    et3 = 0
    et_tlt = 0
    et_rk = []
    r_rk = []
    id = 0
    for ele in sort_nidx:
        if ele in top_news_grd_ele and id < 3:
            et3 += 1
            et_tlt += 1
            et_rk.append(id)
            r_rk.append(rank_ele.index(ele))
        elif ele in top_news_grd_ele and id >= 3:
            et_tlt += 1
        id += 1
    aet3 += et3
    aet_tlt += et_tlt
    et_base += 3

    return sort_nidx, near_news, near_news_sim, all, aet3, aet_tlt, et_base


def MMRScore(Si, center, Sj, lambta, new_rep):
    Sim1 = cosine_similarity(new_rep[Si], center)[0, 0]
    l_expr = lambta * Sim1
    value = [float("-inf")]

    for sent in Sj:
        Sim2 = cosine_similarity(new_rep[Si], new_rep[sent])[0, 0]
        value.append(Sim2)
    r_expr = (1 - lambta) * max(value)
    MMR_SCORE = l_expr - r_expr
    return MMR_SCORE


def MMR(cluster, news_idx, new_rep, clu, core_idx):
    core_in = False
    for ele in core_idx:
        if ele in news_idx:
            core_in = True
            break
    news_cen_dis = {}
    center = cluster.cluster_centers_[clu]
    if core_in == True:
        for ni in news_idx:
            news_cen_dis[ni] = cosine_similarity(new_rep[ni], center)[0, 0]
        max_news_idx = max(news_cen_dis, key=news_cen_dis.get)
        del news_cen_dis[max_news_idx]

        sum_idx = [max_news_idx]
        for num in range(len(news_idx) - 1):
            MMRval = {}
            for ele in news_cen_dis:
                MMRval[ele] = MMRScore(ele, center, sum_idx, 0.5, new_rep)
            max_ele = max(MMRval, key=MMRval.get)
            sum_idx.append(max_ele)
            del news_cen_dis[max_ele]
        for ele in core_idx:
            if ele in sum_idx:
                print '====Core sentence rank is {} of total {}'.format(sum_idx.index(ele), len(sum_idx))
    else:
        print 'Core sentence not in cluster'


def reRank(rank_ele, top_news_grd_ele, top5_num, top5_tlt, tlt, dist, near_news, near_news_sim,
           all, other_clu_dis):
    new_rank = {}
    if len(near_news_sim) > 0:
        min_id = sorted(near_news_sim, key=near_news_sim.get)
        min_value = near_news_sim[min_id[0]]
    else:
        min_value = 0.1
    dist_norm = normalize(dist)[0]

    for m in range(len(rank_ele)):
        ele = rank_ele[m]
        if ele in near_news_sim:
            # score = 0.5* (1-dist[m]) + 0.5 * (near_news[ele]*1.0)/all
            oscore = 0.0
            # for k in range(len(other_clu_dis[m])):
            #     oscore += 0.4*(1-other_clu_dis[m][k])
            score = 0.5 * (1 - dist_norm[m]) + 0.5 * near_news_sim[ele]  # + oscore
            # score = (1-dist[m])*near_news_sim[ele]
            new_rank[ele] = score
        else:
            # score = (1-dist[m])*min_value
            oscore = 0.0
            # for k in range(len(other_clu_dis[m])):
            #     oscore += 0.4 * (1 - other_clu_dis[m][k])
            score = 0.5 * (1 - dist[m]) + 0.5 * min_value  # + oscore
            new_rank[ele] = score

    sort_new_rank = sorted(new_rank, key=new_rank.get, reverse=True)
    rk = 0
    clu5 = 0
    clu_tlt = 0
    for ele in sort_new_rank:
        if rk < 3 and ele in top_news_grd_ele:
            clu5 += 1
            clu_tlt += 1
        elif rk >= 3 and ele in top_news_grd_ele:
            clu_tlt += 1
        rk += 1
    sort_news = []
    if len(sort_new_rank) > 3:
        sort_news = sort_new_rank[:3]
    else:
        sort_news = sort_new_rank
    top5_num += clu5
    top5_tlt += clu_tlt
    tlt += 3

    return top5_num, top5_tlt, tlt, sort_news


def tweet_news_msim(tweet_rep, new_rep, twee_lab, news_lab):
    clu_add_tweet = {}
    for i in range(len(tweet_rep)):
        sim = {}
        for j in range(len(new_rep)):
            # sim[j] = cosine_similarity(new_rep[j], tweet_rep[i])[0, 0]
            mem = [new_rep[j]] + [tweet_rep[i]]
            sim[j] = pairwise_distances(X=mem, metric='euclidean')[0, 1]   #canberra
        # sortn = sorted(sim, key=sim.get, reverse=True)
        sortn = sorted(sim, key=sim.get)
        news_clu = [news_lab[m] for m in [sortn[0]]]
        exist = []
        for k in news_clu:  # set(news_clu)
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


def tweet_clu_avg(tweet_rep, news_rep, label):
    twee_lab = []
    for i in range(len(tweet_rep)):
        clu_sim = []
        for j in range(len(set(label))):
            news_idx = [m for m, x in enumerate(label) if x == j]
            tn_sim = []
            for k in news_idx:
                tn_sim.append(cosine_similarity(tweet_rep[i], news_rep[k])[0, 0])
            avg_tn = sum(tn_sim)/len(tn_sim)
            clu_sim.append(avg_tn)
        max_sim = max(clu_sim)
        twee_lab.append(clu_sim.index(max_sim))
    return twee_lab



def assignTweet(cluster, news_rep, twee_rep, tweets, news_con, news_num, twee_num, summary, rank,
                center, top_news_grd, dist, other_clu_dis, vocab_inv, num3, rest,
                news_grd_sim_value, news_grd_sim_rank, clu_ele, emp_ele, label, noise_emp_ele,
                twee_org, news_org):
    encoder_input = []
    decoder_target = []
    decoder_infer_tar = []
    decoder_infer_pin = []
    vocab_len = len(vocab_inv)
    clu_dir = './km_avgT/'
    if not os.path.exists(clu_dir):
        os.makedirs(clu_dir)
    top5_num = 0
    top5_tlt = 0
    tlt = 0
    aet3 = 0
    aet_tlt = 0
    et_base = 0
    t3_tlt = 0
    all = 0
    t3 = 0
    base = 0
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
        twee_lab = tweet_clu_avg(tweet_rep, new_rep, label)
        # twee_lab, noise = tweet_org_cluster(twee_org[i], news_org[i], label)
        # twee_lab, ele_dis = tweeClu(cluster[i], tweet_rep, ele_idx, vocab_len)
        # twee_lab = tweet_cluster(tweet_rep, center[i])
        clu_add_tweet = tweet_news_msim(tweet_rep, new_rep, twee_lab, label)
        # for j in range(len(set(clu_ele[i]))):
        for j in range(len(set(label))):
            news_idx = [m for m, x in enumerate(label) if x == j]
            twee_idx = [m for m, x in enumerate(twee_lab) if x == j]
            # twee_idx = [m for m, x in enumerate(twee_lab) if ((m not in noise) and ( x == j ))]

            if j in clu_add_tweet:
                twee_idx += clu_add_tweet[j]

            if len(twee_idx) > 0:
                twee_idx_temp = [m + news_num[i] for m in twee_idx]
            else:
                twee_idx_temp = []
            nt_idx = news_idx + twee_idx_temp
            # nt_idx = clu_ele[i][j] + twee_idx_temp
            nt_rep = []
            nt_rep += new_rep
            nt_rep += tweet_rep
            """
            non_noise_idx = []
            for d in range(len(label[i])):
                if label[i][d] == -1:
                    continue
                else:
                    non_noise_idx.append(d)
            text += 'Cluster ' + str(j) + ':\tNews: ' + str(len(clu_ele[i][j])) + '\n'
            text += writeText(news_con[i], clu_ele[i][j], vocab_inv)
            """
            text += 'Cluster ' + str(j) + ':\tNews: ' + str(len(news_idx)) + '\n'
            text += writeText(news_con[i], news_idx, vocab_inv)
            text += writeText(news_con[i], [0, 1], vocab_inv)

            if len(nt_idx) > 1:
                estimator = hdbscan.HDBSCAN(min_cluster_size=2)  # , algorithm='prims_kdtree'
                result = estimator.fit(nt_rep)
                threshold = pd.Series(result.outlier_scores_).quantile(0.8)
                outliers = np.where(result.outlier_scores_ > threshold)[0]

                news_out = [m for m in outliers if m in news_idx]
                # news_out = [m for m in outliers if (m<len(non_noise_idx) and non_noise_idx[m] in clu_ele[i][j])]
                twee_out = [m for m in outliers if m in twee_idx_temp]
                true_twee_out = [m - news_num[i] for m in twee_out]
                center = cluster[i].cluster_centers_
                twee_out_fin = tweet_back(j, news_out, true_twee_out, new_rep, tweet_rep, center)
                twee_idx_fin = [m for m in twee_idx if m not in twee_out_fin]
            else:
                print 'No Element in cluster {} of doc {}'.format(j, i)
                twee_idx_fin = [m - news_num[i] for m in twee_idx_temp]
            """
            sort_nidx, near_news, near_news_sim, all, aet3, aet_tlt, et_base = \
                check_core_news(news_idx, twee_idx_fin, new_rep, tweet_rep, top_news_grd[i],
                                rank[i][j], aet3, aet_tlt, et_base)

            top5_num, top5_tlt, tlt, sort_news = reRank(rank[i][j], top_news_grd[i], top5_num, top5_tlt, tlt,
                                                        dist[i][j], near_news, near_news_sim, all,
                                                        other_clu_dis[i][j])
            """
            temp_n = []
            nt_con = []
            temp_t = []

            if len(news_idx)>0:
                temp_n += concateAspIdx(news_con[i], news_idx, concate=True)
                nt_con += concateAspIdx(news_con[i], news_idx, concate=False)
            """
            if len(clu_ele[i][j]) > 0:
                temp_n += concateAspIdx(news_con[i], clu_ele[i][j], concate=True)
                nt_con += concateAspIdx(news_con[i], clu_ele[i][j], concate=False)
            """
            temp_n += concateAspIdx(news_con[i], [0, 1], concate=True)
            nt_con += concateAspIdx(news_con[i], [0, 1], concate=False)

            if len(twee_idx_fin) > 0:  # twee_idx_fin
                temp_t = concateAspIdx(tweet, twee_idx_fin, concate=False)  #twee_idx_fin
            """
            if len(emp_ele[i][j]) > 0:
                for emp in emp_ele[i][j]:
                    for s in range(3):
                        temp_t.append(news_con[i][emp])
            
            if j in noise_emp_ele[i] > 0:
                for emp in noise_emp_ele[i][j]:
                    for s in range(3):
                        temp_t.append(news_con[i][emp])
            
            news_dis = topCenterTxt(cluster[i], new_rep, news_idx, j, vocab_len)
            news_sort = [m[0] for m in sorted(enumerate(news_dis), key=lambda x: x[1])]
            news_add_idx = [x for m, x in enumerate(news_idx) if m in news_sort]
            if len(news_idx) > 5:
                for s in range(5):
                    temp_t.append(news_con[i][news_add_idx[s]])
            elif len(news_idx) > 0:
                temp_t.append(news_con[i][news_add_idx[0]])

            ci_num = 0
            re = np.unique(rest[i])
            np.random.shuffle(re)
            for ci in re:
                if ci in news_idx and ci_num<3:
                    for s in range(5):
                        temp_t.append(news_con[i][ci])
                    ci_num += 1
                elif ci_num >=3:
                    break
            if ci_num < 3:
                ni = news_idx
                np.random.shuffle(ni)
                count = 0
                while ci_num<3:
                    try:
                        if ni[count] not in num3[i] and count<len(ni):
                            for s in range(5):
                                temp_t.append(news_con[i][ni[count]])
                            count += 1
                            ci_num += 1
                        elif count < (len(ni)-1):
                            count += 1

                        if count >=len(ni)-1:
                            break
                    except:
                        print 'Something is wrong'
                        import pdb; pdb.set_trace()

            ele_rank = {}
            for ele in sort_news:
                rk = min(news_grd_sim_rank[i][ele])
                ridx = news_grd_sim_rank[i][ele].index(rk)
                if rk<3:
                    t3 += 1
                t3_tlt +=1
                si = max(news_grd_sim_value[i][ele])
                sidx = news_grd_sim_value[i][ele].index(si)
                ele_rank[ele] = {rk: ridx, si: sidx}
                for e in range(3):
                    try:
                        temp_t.append(news_con[i][ele])
                    except:
                        print 'index out of range'
                        import pdb; pdb.set_trace()

            print '-- Add news infor: --'
            print ele_rank

            base += 3
            """

            twee_len = len(temp_t)
            encoder = [temp_n] * twee_len
            encoder_input += encoder
            decoder_target += temp_t
            decoder_infer_tar.append(temp_n)
            decoder_infer_pin.append(nt_con)
        cPickle.dump(text, clu_file)
    """
    print 'Total precision of tweets rank is: ', (t3 * 1.0) / base
    print 'Total recall of tweets is: ', (t3 * 1.0) / t3_tlt

    print 'Total precision is: ', (top5_num * 1.0) / tlt
    print 'Total recall is: ', (top5_num * 1.0) / top5_tlt
    """
    return encoder_input, decoder_target, decoder_infer_tar, decoder_infer_pin


def aggClu_core(cluster, news_rep, news_num, core_idx, sum_org, news_org, core_sim):
    for i in range(len(core_idx)):
        sum_num = sum(news_num[:i])
        e_num = sum_num + news_num[i]
        new_rep = news_rep[sum_num:e_num]
        for j in range(len(set(cluster[i].labels_))):
            news_idx = [m for m, x in enumerate(cluster[i].labels_) if x == j]
            """
            core_in = []
            for c in core_idx[i]:
                if c in news_idx:
                    core_in.append(c)

            core_in = sorted(core_in)
            if len(core_in) > 0:
            """
            ele_sim = {}
            # sim_matrix = np.asarray(cluster[i].affinity_matrix_.todense())
            # sim_matrix =
            for ni in news_idx:
                temp_sim = []
                for nei in news_idx:
                    if nei != ni:
                        # temp_sim.append(sim_matrix[ni][nei])
                        # temp_sim.append(cosine_similarity(new_rep[ni], new_rep[nei])[0, 0])
                        temp_sim.append(euclidean_distances(new_rep[ni], new_rep[nei])[0, 0])
                if len(news_idx) > 1:
                    avg_sim = (sum(temp_sim) * 1.0) / len(temp_sim)
                    ele_sim[ni] = avg_sim
                else:
                    ele_sim[ni] = 1
            sort_sim = sorted(ele_sim, key=ele_sim.get, reverse=True)
            cidx = sort_sim[0]
            news_grd_max_sim = news_grd_sim(sum_org[i], news_org[i], [cidx])
            print '\n #### News center similarity with ground truth is ', news_grd_max_sim
            print 'max similarity of news with ground truth is: ', core_sim[i]
            # core_rank = [m for m,x in enumerate(sort_sim) if x in core_in]
            # print '=== Doc {} clu {} core sen {} rank {} total {}'.format(i, j, core_in, core_rank, len(news_idx))
            clu_sim = {}
            for ni in news_idx:
                if ni != cidx:
                    # clu_sim[ni] = sim_matrix[cidx][ni]
                    # clu_sim[ni] = cosine_similarity(new_rep[cidx], new_rep[ni])[0, 0]
                    clu_sim[ni] = euclidean_distances(new_rep[cidx], new_rep[ni])[0, 0]
            core_sen_sim = sorted(clu_sim, key=clu_sim.get, reverse=True)
            news_grd_max_sim = news_grd_sim(sum_org[i], news_org[i], core_sen_sim)
            print '== News sentence ranking to cidx is ', core_sen_sim
            print '** News sentence similarity with ground truth'
            print news_grd_max_sim
            # core_sim_rank = [m for m,x in enumerate(core_sen_sim) if x in core_in]
            # print '** clu{} top ind is {}'.format(j, cidx)
            # print '*** core sen rank {} to centern sen of {}'.format(core_sim_rank, len(news_idx))
            # else:
            #     print '+++ No core sen in doc {} clu {}'.format(i, j)


def news_grd_sim(sum_org, news_org, news_idx):
    news_grd_max_sim = {}
    for n in news_idx:
        max_sim = 0
        for s in range(len(sum_org)):
            sim = sm(None, sum_org[s], news_org[n]).ratio()
            if sim > max_sim:
                max_sim = sim
        news_grd_max_sim[n] = max_sim
    return news_grd_max_sim


def sum_news_sen(sum_org, news_org, news_con, summary):
    news_idx = []
    extr_sum = []
    reference = []
    core_sim = []
    for i in range(len(sum_org)):
        max_news_idx = []
        temp_sum = []
        sum_con = []
        con = []
        temp_max = []
        for j in range(len(sum_org[i])):
            sim_ratio = []
            for k in range(len(news_org[i])):
                sim_ratio.append(sm(None, sum_org[i][j], news_org[i][k]).ratio())
            idx = [m[0] for m in sorted(enumerate(sim_ratio), key=lambda x: x[1], reverse=True)]
            mx = max(sim_ratio)
            temp_max.append(mx)
            for m in range(len(idx)):
                if idx[m] not in max_news_idx:
                    max_news_idx.append(idx[m])
                    temp_sum += news_con[i][idx[m]]
                    break
            con += summary[i][j]
        # print 'Most rep news sens for doc {} is {}'.format(i, max_news_idx)
        core_sim.append(temp_max)
        sum_con.append(con)
        reference.append(sum_con)
        extr_sum.append(temp_sum)
        news_idx.append(max_news_idx)
    return news_idx, extr_sum, reference, core_sim


"""
if __name__=='__main__':
    embed_dim = 300
    batchsize = 128
    pre_embed = []
    restore = True

    summary, news, news_twee, tweets, vocab, vocab_inv, max_twee_len, news_con, title, first, \
        twee_num, sum_org, news_org = loop_dir()
    news_idx, extr_sum, reference, core_sim = sum_news_sen(sum_org, news_org, news_con, summary)
    # rouge(extr_sum, reference)
    vocab_len = len(vocab_inv)

    news2, news_num = conv3to2(news_con)
    s2s = unsuS2S_adv(vocab_len, embed_dim, batchsize, pre_embed, attention=True)

    with s2s.graph.as_default():
        encoder_input, en_in_len, encoder_output, encoder_final_state = s2s.encoder()
        de_out, de_out_len, title_out, first_out, decoder_logits, decoder_prediction, loss_weight, \
        decoder_target, decoder_title, decoder_first, decoder_prediction_inference \
            = s2s.decoder_adv(max_twee_len)

        saver = tf.train.Saver()
        if restore:
            saver.restore(s2s.sess, './log2/pre_check_point-99')
        else:
            tf.set_random_seed(1)
            init = tf.global_variables_initializer()
            s2s.sess.run(init)

        feed_dict = next_feed_inference(news2, encoder_input, en_in_len)
        news_state = s2s.sess.run(encoder_final_state, feed_dict)
        sen_rep = news_state.h
        news_rep = sen_rep.tolist()

        feed_dict = next_feed_inference(tweets, encoder_input, en_in_len)
        twee_state = s2s.sess.run(encoder_final_state, feed_dict)
        sen_rep = twee_state.h
        twee_rep = sen_rep.tolist()

        # news_twee_con, news_num = prepareData(news_con, tweets, twee_num)
        # news_tfidf, twee_tfidf = news_twee_tfidf(news_twee_con, news_num, vocab_len)
        # news_rep, news_num = conv3to2(news_tfidf)
        # twee_rep, twee_num = conv3to2(news_tfidf)
        cluster = cluster_demo(news_rep, news_num, summary, news_idx)
        print 'Clustering Done'
        top_news_grd = grd_news(sum_org, news_org)
        rank = top_quality(cluster, news_rep, news_num, top_news_grd, sum_org, news_org)
        # aggClu_core(cluster, news_rep, news_num, news_idx, sum_org, news_org, core_sim)
        encoder_input, decoder_target, decoder_infer_tar, decoder_infer_pin \
            = assignTweet(cluster, news_rep, twee_rep, tweets, news_con, news_num,
                          twee_num, summary, vocab_inv, rank, top_news_grd)
"""