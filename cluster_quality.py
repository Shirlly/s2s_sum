from difflib import SequenceMatcher as sm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
from kmeans_staff1 import tweeClu
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd



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


def re_clu_center(idx, rep, top_news_grd_ele, top5, tlt, base):
    ele_sim = {}
    for ni in idx:
        temp_sim = []
        for nei in idx:
            if nei != ni:
                # temp_sim.append(sim_matrix[ni][nei])
                temp_sim.append(cosine_similarity(rep[ni], rep[nei])[0, 0])
                # temp_sim.append(euclidean_distances(new_rep[ni], new_rep[nei])[0, 0])
        if len(idx) > 1:
            avg_sim = (sum(temp_sim) * 1.0) / len(temp_sim)
            ele_sim[ni] = avg_sim
        else:
            ele_sim[ni] = 1
    sort_sim = sorted(ele_sim, key=ele_sim.get, reverse=True)
    # cidx = sort_sim[0]
    clu_tlt = 0
    clu_top5 = 0
    for m in range(len(sort_sim)):
        ele = sort_sim[m]
        if ele in top_news_grd_ele and m < 5:
            clu_top5 += 1
            clu_tlt += 1
        elif ele in top_news_grd_ele and m >= 5:
            clu_tlt += 1
    top5 += clu_top5
    tlt += clu_tlt
    base += 5
    print '== Cluster based precision is ', (clu_top5 * 1.0) / 5
    if clu_tlt > 0:
        print '== Cluster based recall is ', (clu_top5 * 1.0) / clu_tlt
    else:
        print '== Cluster based recall is 0'
    return top5, tlt, base, sort_sim


def avg_clu_center(idx, rep):
    add_rep = np.asarray(rep[idx[0]])
    for i in range(1, len(idx)):
        add_rep += np.asarray(rep[idx[i]])
    center = add_rep/len(idx)
    return center


def top_quality(cluster, news_rep, news_num, top_news_grd, sum_org, news_org):
    tlt = 0
    top5 = 0
    base = 0
    rank = []
    sim = []
    clu_center = []
    other_clu_dis = []
    for i in range(len(sum_org)):
        sum_num = sum(news_num[:i])
        e_num = sum_num + news_num[i]
        new_rep = news_rep[sum_num:e_num] # news_rep[i]
        crank = []
        csim = []
        ccenter = []
        other_dis = []
        for j in range(len(set(cluster[i].labels_))):
            news_idx = [m for m, x in enumerate(cluster[i].labels_) if x == j]

            center = avg_clu_center(news_idx, new_rep)
            # center = cluster[i].cluster_centers_[j]
            cidx = -1
            top5, tlt, base, sort_news_id, news_sim, other_sort = cluster_top5_grd_rank(news_idx, cidx, new_rep,
                                                                            top_news_grd[i], top5, tlt,
                                                                            base, center, cluster[i], j)
            # top5, tlt, base, core_sen_sim = MMR(center, news_idx, new_rep, top_news_grd[i], top5, tlt, base)
            # top5, tlt, base, core_sen_sim = re_clu_center(news_idx, new_rep, top_news_grd[i], top5, tlt, base)
            # news_grd_max_sim = news_grd_sim(sum_org[i], news_org[i], core_sen_sim[:5])
            # rk = [cidx] + core_sen_sim
            rk = sort_news_id
            other_dis.append(other_sort)
            crank.append(rk)
            csim.append(news_sim)
            ccenter.append(center)
        rank.append(crank)
        sim.append(csim)
        clu_center.append(ccenter)
        other_clu_dis.append(other_dis)
    print '++ Total precision is ', (top5*1.0)/(base)
    print '++ Total recall is ', (top5*1.0)/(tlt)
    return rank, clu_center, sim, other_clu_dis

def cluster_top5_grd_rank(news_idx, cidx, new_rep, top_news_grd, top5, tlt, base, center, cluster, clu):
    clu_sim = []
    other = []
    for ni in news_idx:
        if ni != cidx:
            # clu_sim[ni] = sim_matrix[cidx][ni]
            # clu_sim[ni] = cosine_similarity(new_rep[cidx], new_rep[ni])[0, 0]
            ele = [center.tolist()] + [new_rep[ni]]
            dis = pairwise_distances(ele, metric="euclidean")[0,1]
            # dis = cosine_similarity(center, new_rep[ni])[0, 0]
            # dis = cluster.transform(new_rep[ni], [1, -1])[0]
            clu_sim.append(dis)   #dis[clu]
            """
            norm_dis = normalize(dis)[0]
            other_dis = []
            for k in range(len(set(cluster.labels_))):
                if k != clu:
                    other_dis.append(norm_dis[k])
            other.append(other_dis)
            """
            # clu_sim[ni] = euclidean_distances(new_rep[cidx], new_rep[ni])[0, 0]
    # core_sen_sim = sorted(clu_sim, key=clu_sim.get, reverse=True)

    id = [m[0] for m in sorted(enumerate(clu_sim), key=lambda x: x[1])]   # , reverse=True
    news_sim = [m[1] for m in sorted(enumerate(clu_sim), key=lambda x: x[1])]  #, reverse=True
    clu_tlt = 0
    clu_top5 = 0
    sort_news_id = []
    other_sort = []
    for m in range(len(id)):
        ele = news_idx[id[m]]
        sort_news_id.append(ele)
        # other_sort.append(other[id[m]])
        if ele in top_news_grd and m < 3:
            clu_top5 += 1
            clu_tlt += 1
        elif ele in top_news_grd and m >= 3:
            clu_tlt += 1
    top5 += clu_top5
    tlt += clu_tlt
    base += 3
    """
    print '== Cluster based precision is ', (clu_top5 * 1.0) / 3
    if clu_tlt > 0:
        print '== Cluster based recall is ', (clu_top5 * 1.0) / clu_tlt
    else:
        print '== Cluster based recall is 0'
    """
    return top5, tlt, base, sort_news_id, news_sim, other_sort


def tweet_cluster(twee_rep, center):
    tweet_lab = []
    for i in range(len(twee_rep)):
        tc_sim = {}
        for j in range(len(center)):
            # tc_sim[j] = cosine_similarity(center[j], twee_rep[i])[0, 0]
            mem = [center[j].tolist()] + [twee_rep[i]]
            tc_sim[j] = pairwise_distances(X=mem, metric='euclidean')[0, 1]   # canberra
        clu_rank = sorted(tc_sim, key=tc_sim.get, reverse= True)
        """ Soft ranking """
        tweet_lab.append(clu_rank[0])
    return tweet_lab


def tweet_org_cluster(twee_org, news_org, label):
    twee_lab = []
    sim = []
    for i in range(len(twee_org)):
        tn_sim = {}
        for j in range(len(news_org)):
            if label[j] != -1:
                tn_sim[j] = sm(None, twee_org[i], news_org[j]).ratio()
        rank = sorted(tn_sim, key=tn_sim.get, reverse=True)
        twee_lab.append(label[rank[0]])
        sim.append(tn_sim[rank[0]])
    sim = np.asarray(sim)
    threshold = pd.Series(sim).quantile(0.2)
    noise = np.where(sim < threshold)[0]
    return twee_lab, noise


def MMRScore(Si, center, Sj, lambta, new_rep):
    # Sim1 = cosine_similarity(new_rep[Si], center)[0, 0]
    mem = [center.tolist()] + [new_rep[Si]]
    Sim1 = pairwise_distances(X=mem, metric='euclidean')[0, 1]  #canberra

    l_expr = lambta * Sim1
    value = [float("-inf")]

    for sent in Sj:
        # Sim2 = cosine_similarity(new_rep[Si], new_rep[sent])[0, 0]
        mem = [new_rep[sent]] + [new_rep[Si]]
        Sim2 = pairwise_distances(X=mem, metric='euclidean')[0, 1]  #canberra
        value.append(Sim2)
    r_expr = (1 - lambta) * max(value)
    MMR_SCORE = l_expr - r_expr
    return MMR_SCORE


def MMR(center, news_idx, new_rep, top_news_grd, top5, tlt, base):
    news_cen_dis = {}
    for ni in news_idx:
        # news_cen_dis[ni] = cosine_similarity(new_rep[ni], center)[0, 0]
        mem = [center.tolist()] + [new_rep[ni]]
        news_cen_dis[ni] = pairwise_distances(X=mem, metric='euclidean')[0,1]  #canberra
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

    clu_tlt = 0
    clu_top5 = 0
    for m in range(len(sum_idx)):
        ele = sum_idx[m]
        if ele in top_news_grd and m < 5:
            clu_top5 += 1
            clu_tlt += 1
        elif ele in top_news_grd and m >= 5:
            clu_tlt += 1
    top5 += clu_top5
    tlt += clu_tlt
    base += 5
    """
    print '== Cluster based precision is ', (clu_top5 * 1.0) / 5
    if clu_tlt > 0:
        print '== Cluster based recall is ', (clu_top5 * 1.0) / clu_tlt
    else:
        print '== Cluster based recall is 0'
    """
    return top5, tlt, base, sum_idx



def tweet_grd(tweet_org, twee_num, sum_org, news_grd_sim_value, news_grd_sim_rank):
    twee_grd_sim_value = []
    twee_grd_sim_rank = []
    top3_tweet = []

    tlt_ts_num = 0
    tlt_ns_num = 0
    tlt_ts_max = 0
    tlt_ns_max = 0
    for i in range(len(sum_org)):
        # snum = sum(twee_num[:i])
        # enum = snum + twee_num[i]
        tweet = tweet_org[i]
        asp_ts_sim = []
        asp_ts_rank = []
        most_sim_twee = {}
        ts_num = 0
        ns_num = 0
        ts_max_num = 0
        ns_max_num = 0
        for s in sum_org[i]:
            ts_sim = []
            for t in tweet:
                ts_sim.append(sm(None, s, t).ratio())
            asp_ts_sim.append(ts_sim)
            idx = [m[0] for m in sorted(enumerate(ts_sim), key=lambda x: x[1], reverse=True)]
            sort_sim = [m[1] for m in sorted(enumerate(ts_sim), key=lambda x: x[1], reverse=True)]
            sim_rank = []
            for id in range(len(idx)):
                sim_rank.append(idx.index(id))
            asp_ts_rank.append(sim_rank)
            if len(sort_sim)>0:
                ts_max = sort_sim[0]
            else:
                import pdb; pdb.set_trace()
            for k in range(len(sort_sim)):
                if sort_sim[k] >= 0.5:
                    ts_num += 1
                    """
                    if idx[k] in most_sim_twee:
                        num = most_sim_twee[idx[k]]
                        most_sim_twee[idx[k]] = num + 1
                    else:
                        most_sim_twee[idx[k]] = 1
                    """
                else:
                    break
            ns_max = 0.0
            for n in range(len(news_grd_sim_value[i])):
                ns_sim = news_grd_sim_value[i][n]
                if max(ns_sim) > ns_max:
                    ns_max = max(ns_sim)
                if max(ns_sim) > 0.5:
                    ns_num += 1
            if ns_max > ts_max:
                ns_max_num += 1
            else:
                ts_max_num += 1

        print 'NS similarity larger than 0.5 number is ', ns_num
        print 'TS similarity larger than 0.5 number is ', ts_num
        # print 'Tweets > news ', ts_max_num
        # print 'News > tweets', ns_max_num
        tlt_ts_num += ts_num
        tlt_ns_num += ns_num
        tlt_ts_max += ts_max_num
        tlt_ns_max += ns_max_num
        new_asp_sim = map(list, zip(*asp_ts_sim))
        twee_grd_sim_value.append(new_asp_sim)
        new_asp_rank = map(list, zip(*asp_ts_rank))
        twee_grd_sim_rank.append(new_asp_rank)
    print '++ Total NS similarity larger than 0.5 number is ', tlt_ns_num
    print '++ Total TS similarity larger than 0.5 number is ', tlt_ts_num
    print '++ Total Tweets > news ', tlt_ts_max
    print '++ Total News > tweets', tlt_ns_max
    return twee_grd_sim_value, twee_grd_sim_rank


def news_tweets(news_org, tweet_org, news_grd_sim_rank, twee_grd_sim_rank):
    rk5 = 0
    tlt_rk5 = 0
    top5_rk = []
    for i in range(len(news_org)):
        nt_sim_num = {}
        news_id = 0
        for n in news_org[i]:
            nt_sim = []
            for t in tweet_org[i]:
                nt_sim.append(sm(None, n, t).ratio())
            # idx = [m[0] for m in sorted(enumerate(nt_sim), key=lambda x: x[1], reverse=True)]
            # sort_sim = [m[1] for m in sorted(enumerate(nt_sim), key=lambda x: x[1], reverse=True)]
            large_num = len([m for m, x in enumerate(nt_sim) if x > 0.6])
            nt_sim_num[news_id] = large_num
            news_id += 1
        nidx = sorted(nt_sim_num, reverse=True)
        news_rank = news_grd_sim_rank[i]
        for j in range(5):
            min_rk = min(news_rank[nidx[j]])
            if min_rk < 5:
                rk5 += 1
        tlt_rk5 += 5
    print 'Top 5 news sentence Ranked by tweets number ', (1.0 * rk5)/tlt_rk5

