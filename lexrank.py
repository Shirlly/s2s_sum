from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from lex_rank import LexRankSummarizer
import os
import argparse
import numpy as np
import tensorflow as tf
from data_add import loop_dir
import cPickle
from helpers import batch
from unsupervisedS2Sadv1 import unsuS2S_adv
from extractSummary import rouge
from cluster_process import cluster_demo, grd_news
from difflib import SequenceMatcher as sm
from preprare_cluster_data import conv2to3, conv3to2, next_feed_inference
from sklearn.metrics.pairwise import cosine_similarity
from preEmbedding import w2v1


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", default=64)
parser.add_argument("-d", "--embed_dim", default=300)
parser.add_argument("-l", "--learning_rate", default=0.001)
parser.add_argument("-e", "--max_epoch", default=100)
parser.add_argument("-cr", "--cluster_rep_restore", default=True)
parser.add_argument("-crd", "--cluster_rep_dir", default='./pre_log_Adam_con')
parser.add_argument("-mr", "--model_restore", default=True)
parser.add_argument("-mrd", "--model_dir", default='./pre_log_Adam_con')
parser.add_argument("-m", "--cluster_method", default='kmeans')
parser.add_argument("-wh", "--word_hidden_dimension", default=128)
parser.add_argument("-sh", "--sen_hidden_dimension", default=128)
parser.add_argument("-c", "--cell", default='GRU')
parser.add_argument("-a", "--attention_level", default='sen')
parser.add_argument("-f", "--log_root", default='./log')
parser.add_argument("--mode", default='train')
parser.add_argument("--decode_mode", default='sen_out', help='sen_out: attention on sentence_encoder_output; '
                                                             'word_out: attention on word_encoder_output')
parser.add_argument("--beam_size", default=4)
parser.add_argument("--alpha", default=0.8, help='Weight for tweet vote score')
parser.add_argument("--max_grad_norm", default=2.0, help='for gradient clipping')
parser.add_argument("--adagrad_init_acc", default=0.1, help='initial accumulator value for Adagrad')
parser.add_argument("--opt", default='Adagrad', help='Optimization method, Adam or Adagrad')
parser.add_argument("--opt_clip", default=True, help='Apply gradient clip on the optimizer')
parser.add_argument("--valid", default=True, help='Adopt valid training mechanism')
parser.add_argument("--best_epoch_path", default='./log_Adam_True_False_plain/best_epoch_checkpoint-1', help='Path for best model on epoch loss')
parser.add_argument("--train_dir", default='', help='Directory storing various data')
parser.add_argument("--all", default=True, help='Retrain the model by all the data or only extracted news_text')
parser.add_argument("--encoder_dropout", default=True, help='Encoder with dropout')
parser.add_argument("--decoder_dropout", default=True, help='Decoder with dropout')
args = parser.parse_args()




def convert_title_uppercase(news_org):
    file_dir = './news_text/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    for i in range(len(news_org)):
        # print 'filename ', i
        text = ''
        for j in range(len(news_org[i])):
            line = news_org[i][j].lower()
            # if j == 0:
            #     line = line.upper()
            text += line + '\n\n'
        write_path = os.path.join(file_dir, str(i))
        file_out = open(write_path, 'w')
        file_out.write(text)
        file_out.close()


def tweet_vote(news_con):
    data = cPickle.load(open('./epoch_tweet_news_all', 'rb'))
    news_idx_clu = data['news_idx_clu']
    avg_weight_dict = data['avg_weight_dict']
    """
    tweet_attn = {}
    for key in news_idx_clu.keys():
        doc_id, clu_id = key.split(',')
        doc_id = int(doc_id)
        if doc_id in tweet_attn:
            attn = tweet_attn[doc_id]
        else:
            attn = {}
        news_id = news_idx_clu[key]
        for i in range(len(news_id)):
            id = news_id[i]
            attn[id] = avg_weight_dict[key][0][i]
        tweet_attn[doc_id] = attn
    """
    p_vector = []
    num = 0
    for i in range(len(news_con)):  # tweet_attn
        attn = avg_weight_dict[i]  # tweet_attn[i]
        doc_len = len(news_con[i])
        sort_attn = np.zeros([doc_len], dtype='int32')
        for j in range(doc_len):
            if j in attn:
                sort_attn[j] = attn[j]
        p_vector.append(sort_attn)
    print "Empty sen num ", num
    return p_vector


def cluster_lexrank(news_con, news_org, news_grd_sim_rank, news3):
    # , data_info, train_encoder_in, train_decodertar
    data = cPickle.load(open('./epoch_tweet_news_Adam', 'rb'))
    news_idx_clu = data['news_idx_clu']
    avg_weight_dict = data['avg_weight_dict']
    top3 = 0
    tlt = 0
    base = 0
    short = 0
    for key in news_idx_clu.keys():
        doc_id, clu_id = key.split(',')
        doc_id = int(doc_id)
        news_id = news_idx_clu[key]
        p_vector = avg_weight_dict[key][0]

        # news_rep = news3[doc_id][news_id]
        # matrix = cosine_similarity(news_rep)
        """
        sort_id = [m[0] for m in sorted(enumerate(p_vector), key=lambda x: x[1], reverse=True)]
        top_attn_id = sort_id[:3] if len(sort_id) > 3 else sort_id
        attn_rk = []
        nid = []
        for e in top_attn_id:
            rk = min(news_grd_sim_rank[doc_id][news_id[e]])
            nid.append(news_id[e])
            attn_rk.append(rk)
            if rk < 3:
                top3 += 1
        tlt += 3
        base += 3
        print "Attention select top news {} ranking {}".format(nid, attn_rk)
        """
        if len(news_id) != len(p_vector):
            print "Length wrong!!!"
        abs_dir = './text'
        text = ''
        clu_text = []
        for i in range(len(news_id)):
            id = news_id[i]
            text += news_org[doc_id][id].lower() + '\n\n'
            clu_text.append(news_org[doc_id][id].lower())
        
        file_out = open(abs_dir, 'w')
        file_out.write(text)
        file_out.close()
        matrix = []
        parser = PlaintextParser.from_file(abs_dir, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary, score = summarizer(parser.document, 4, p_vector, matrix)  # Summarize the document with 5 sentences
        new_score_dict = {str(key): score[key] for key in score}
        sort_dict = sorted(new_score_dict, key=new_score_dict.get, reverse=True)

        lower_news = []
        for news in news_org[doc_id]:
            lower_news.append(news.lower())
        top_rk = []
        for sen in summary:
            sen = str(sen)
            ele = lower_news.index(sen)
            rk = min(news_grd_sim_rank[doc_id][ele])
            top_rk.append(rk)
            if rk < 3:
                top3 += 1
        tlt += 3
        base += 3
        # print "Ranking for selected news ", top_rk
        """
        pred_sum.append(pred_temp)
        ref.append(ref_temp)
        reference.append(ref)
        """
    print "== Precision of Lexrank for top 3 news is ", (top3 * 1.0) / tlt
    print "== Recall of Lexrank for top 3 news is ", (top3 * 1.0) / base


def semantic_rep(vocab_size, embed_dim, batch_size, pre_embed, args):
    news2, news_num = conv3to2(news_con)
    tweet3 = conv2to3(tweets, twe_num, news_con)
    s2s = unsuS2S_adv(vocab_size, embed_dim, batch_size, pre_embed, attention=True)

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
        """
        feed_dict = next_feed_inference(tweets, encoder_input, en_in_len)
        twee_state = s2s.sess.run(encoder_final_state, feed_dict)
        sen_rep = twee_state  # twee_state.h  #
        twee_rep = sen_rep.tolist()
        twee3 = conv2to3(twee_rep, twe_num, news_con)
        """
    return news3


def sum_news(news_org, news_grd_sim_rank, grd_summary, news_con, summary_con, news3):
    file_dir = './news_text/'
    top3 = 0
    tlt = 0
    base = 0
    lex_rank_score = []
    pred_sum = []
    reference = []
    p_vector = tweet_vote(news_con)
    tlt_per = []
    tlt_cluster = []
    for filename in sorted(os.listdir(file_dir), key=int):
        # print 'filename ', filename
        i = int(filename)
        abs_dir = os.path.join(file_dir, filename)
        avg_p = np.divide(np.asarray(p_vector[i]), sum(p_vector[i]))
        # news_rep = news3[i]
        # matrix = cosine_similarity(news_rep)
        # matrix[matrix < 0.9] = 0
        matrix = []
        parser = PlaintextParser.from_file(abs_dir, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary, score = summarizer(parser.document, len(grd_summary[i]), avg_p, matrix)
        new_score = {str(key): score[key] for key in score}

        rk_score = []
        for ele in news_org[i]:
            ele = ele.lower()
            try:
                if ele in new_score:
                    rk_score.append(new_score[ele])
            except:
                import pdb; pdb.set_trace()
        lex_rank_score.append(rk_score)
        sen_id = []
        pred_temp = []
        ref_temp = []
        ref = []
        num = 0

        rk_li = np.asarray(news_grd_sim_rank[i])
        top_idx = []
        for k in range(rk_li.shape[1]):
            idx = np.where(rk_li[:, k] < 3)[0].tolist()
            top_idx += idx
        lower_news = []
        for news in news_org[i]:
            lower_news.append(news.lower())

        for sentence in summary:
            sentence = str(sentence)
            # pred_temp.append(sentence)
            # ref_temp.append(grd_summary[i][num])
            news_id = lower_news.index(sentence)

            pred_temp += news_con[i][news_id]
            ref_temp += summary_con[i][num]
            num += 1

            rk = min(news_grd_sim_rank[i][news_id])
            if rk < 3:
                top3 += 1
        per = news_range(lower_news, news_grd_sim_rank[i], news_con[i], i, new_score, grd_summary[i])
        tlt_per.append(per)
        pred_sum.append(pred_temp)
        ref.append(ref_temp)
        reference.append(ref)
        tlt += len(grd_summary[i])
        base += 3 * len(news_grd_sim_rank[i][0])
    print "== Precision of Lexrank for top 3 news is ", (top3 * 1.0) / tlt
    print "== Recall of Lexrank for top 3 news is ", (top3 * 1.0) / base
    print "Total percentage of news is ", sum(tlt_per)/len(tlt_per)
    # cPickle.dump(lex_rank_score, open('./lex_rank_score', 'wb'))
    return lex_rank_score, pred_sum, reference


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
    for sentence in sort_score:
        if len(cluster) < len(summary):
            sentence = str(sentence)
            news_id = lower_news.index(sentence)
            id_range = []
            if news_id not in clu_ele:
                if news_id - 3 <= 0:
                    id_range = range(0, 6)
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
                    clu_ele += id_range
                    num += 1

    # print "Doc {} top_idx: {}   id_range: {}   tlt_inrange: {}".\
    #     format(i, set(top_idx), cluster, set(clu_ele))
    for nid in inter:
        inter_len = {}
        for clu in cluster:
            in_len = len(list(set(inter[nid]).intersection(set(cluster[clu]))))
            inter_len[clu] = in_len
        sort_inter = sorted(inter_len, key=inter_len.get, reverse=True)
        clu = sort_inter[0]
        cluster[clu].append(nid)

    clu_ele += [0, 1]
    common = list(set(top_idx).intersection(set(clu_ele)))
    per = (len(common) * 1.0) / len(set(top_idx))
    print "Percentage {} has been included in selected parts".format(per)
    # print "Elements in selected clusters ", set(clu_ele)
    print "Top news id ", top_idx
    print "Document length ", len(score)
    print "Cluster ", cluster
    return per


if __name__ == '__main__':
    summary, news, news_twee, tweets, vocab, vocab_inv, max_twee_len, news_con, title, first, \
    twe_num, news_num1, sum_org, news_org, tweet_org = loop_dir()
    top_news_grd, num3, rest, news_grd_sim_value, news_grd_sim_rank = grd_news(sum_org, news_org)
    pre_embed, _ = w2v1(vocab)
    news3 = semantic_rep(len(vocab_inv), args.embed_dim, args.batch_size, pre_embed, args)
    # news3 = []
    # cluster_lexrank(news_con, news_org, news_grd_sim_rank, news3)

    convert_title_uppercase(news_org)
    lex_rank_score, pred_sum, reference = sum_news(news_org, news_grd_sim_rank, sum_org, news_con, summary, news3)
    print 'Concatenate summary:'
    result_cat = rouge(pred_sum, reference)

