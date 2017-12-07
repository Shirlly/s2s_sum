from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, \
    AffinityPropagation, DBSCAN
from difflib import SequenceMatcher as sm


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


def cluster_demo(news_rep, news_num, summary):
    cluster = []
    for i in range(len(news_num)):
        clu_num = len(summary[i])

        tnews = news_rep[i]
        if len(tnews) > 40:
            tnews = tnews[:40]
        clu = 'kmeans'
        result = cluster_choice(tnews, clu, clu_num)
        cluster.append(result)
    return cluster

def grd_news(sum_org, news_org):
    top_news_grd = []
    num3 = []
    rest = []
    news_grd_sim_value = []
    news_grd_sim_rank = []
    for i in range(len(sum_org)):
        top_sim_dict = {}
        n3 = []
        re = []
        asp_sim = []
        asp_rank = []
        for j in range(len(sum_org[i])):
            sim_ratio = []
            for k in range(len(news_org[i])):
                sim_ratio.append(sm(None, sum_org[i][j], news_org[i][k]).ratio())
            asp_sim.append(sim_ratio)
            idx = [m[0] for m in sorted(enumerate(sim_ratio), key=lambda x: x[1], reverse=True)]
            sort_sim = [m[1] for m in sorted(enumerate(sim_ratio), key=lambda x: x[1], reverse=True)]
            sim_rank = []
            for id in range(len(idx)):
                sim_rank.append(idx.index(id))
            asp_rank.append(sim_rank)
            idx_fin = []
            n3.append(idx[:2])
            for m in idx[-5:]:
                re.append(idx[m])
            for s in range(len(sort_sim)):
                if sort_sim[s] > 0.5:
                    idx_fin.append(idx[s])
            if len(idx_fin)<5:
                idx_fin = idx[:5]
            else:
                print 'Number of sim news larger than 0.5 is ', len(idx_fin)
            top_num = len(idx_fin)
            for m in range(top_num):
                if idx_fin[m] not in top_sim_dict:
                    top_sim_dict[idx_fin[m]] = 1
                else:
                    num = top_sim_dict[idx_fin[m]]
                    top_sim_dict[idx_fin[m]] = num + 1
        # print 'The least similar news sentence length is ', len(set(re))
        new_asp_sim = map(list, zip(*asp_sim))
        news_grd_sim_value.append(new_asp_sim)
        new_asp_rank = map(list, zip(*asp_rank))
        news_grd_sim_rank.append(new_asp_rank)
        top_news_grd.append(top_sim_dict)
        num3.append(n3)
        rest.append(re)
    return top_news_grd, num3, rest, news_grd_sim_value, news_grd_sim_rank
