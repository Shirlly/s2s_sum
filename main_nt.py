import os
import math
import argparse
import cPickle
import collections
import numpy as np
import tensorflow as tf
import tensorgraph as tg
from data_add import loop_dir
from preEmbedding import w2v1
from random import shuffle
from unsupervisedS2Sadv2 import unsuS2S_adv
from cluster_process import cluster_demo, grd_news
from preprare_cluster_data import train_data
from helpers import word_encoder_batch, batch
from sklearn.preprocessing import normalize
from extractSummary import rouge, cosineSim, cosineSimTop
from piece_input_data import piece_data

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
parser.add_argument("-s", "--prep", default=False, help='Apply tweet attn as p_vector or not')
parser.add_argument("--mode", default='train')
# parser.add_argument("--decode_mode", default='sen_out', help='sen_out: attention on sentence_encoder_output; '
#                                                              'word_out: attention on word_encoder_output')
parser.add_argument("--beam_size", default=4)
parser.add_argument("--alpha", default=0.8, help='Weight for tweet vote score')
# parser.add_argument("--max_grad_norm", default=2.0, help='for gradient clipping')
# parser.add_argument("--adagrad_init_acc", default=0.1, help='initial accumulator value for Adagrad')
parser.add_argument("--opt", default='Adagrad', help='Optimization method, Adam or Adagrad')
parser.add_argument("--opt_clip", default=True, help='Apply gradient clip on the optimizer')
parser.add_argument("--valid", default=True, help='Adopt valid training mechanism')
parser.add_argument("--best_epoch_path", default='./log_Adam_True_False_plain/best_epoch_checkpoint-1', help='Path for best model on epoch loss')
parser.add_argument("--train_dir", default='', help='Directory storing various data')
#parser.add_argument("--all", default=True, help='Retrain the model by all the data or only extracted news_text')
parser.add_argument("--all", default="avg", help='average or sum as hidden layer of encoder and decoder')
parser.add_argument("--encoder_dropout", default=True, help='Encoder with dropout')
parser.add_argument("--decoder_dropout", default=True, help='Decoder with dropout')
parser.add_argument("--multi_layer", default=False, help='Multi RNN layers for encoder')
args = parser.parse_args()

PAD = 0


def simple_cost(decoder_train_targets, train_logits, vocab_size):
    de_tar = tf.transpose(decoder_train_targets, [1, 0])
    label_in = tf.one_hot(de_tar, depth=vocab_size, dtype='float32')
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=label_in,
        logits=train_logits
    )
    _loss = tf.reduce_mean(stepwise_cross_entropy)
    tf.summary.scalar('loss', _loss)
    if args.opt == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(args.learning_rate, initial_accumulator_value=args.adagrad_init_acc)
    elif args.opt == 'Adam':
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
    elif args.opt == 'RMS':
        optimizer = tf.train.RMSPropOptimizer(args.learning_rate)

    if args.opt_clip == True:
        tvars = tf.trainable_variables()
        gradients = tf.gradients(_loss, tvars)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1)
        tf.summary.scalar('global_norm', global_norm)
        _train_op = optimizer.apply_gradients(zip(grads, tvars), name='train_step')
    else:
        _train_op = optimizer.minimize(_loss)

    return _loss, _train_op


def to_text(idx, vocab_inv):
    text = ""
    for word in idx:
        if word in vocab_inv:
            text = text + " " + vocab_inv[word]
        else:
            text = text + ' UNK'
    return text.strip()


def run_training(s2s, train_encoder_in, train_decoder_tar, train_lead, valid_encoder_in, valid_decoder_tar, vocab_inv,
                 max_twee_len, train_dir, summary, news_con):
    with s2s.graph.as_default():
        encoder_input, en_in_len, encoder_output, encoder_final_state = s2s.encoder()
        de_out, de_out_len, decoder_logits, decoder_prediction, loss_weight, \
            decoder_target, decoder_prediction_inference, lead, lead_out_len, decoder_lead, \
            decoder_outputs_train, decoder_state_train=\
            s2s.decoder_adv(max_twee_len)
        vocab_size = len(vocab_inv)
        # loss, train_op = simple_cost(decoder_target, decoder_logits, vocab_size)
        loss, train_op, decoder_mask = s2s.multi_cost(args)
        train_cost = tf.summary.scalar('train_cost', loss)
        train_cost_his = tf.summary.histogram("histogram_cost", loss)
        summ_op = tf.summary.merge_all()

        saver = tf.train.Saver()
        if args.model_restore == True:
            saver.restore(s2s.sess, './log2_gru_embed21/pre_check_point-49')
        else:
            tf.set_random_seed(1)
            init = tf.global_variables_initializer()
            s2s.sess.run(init)

        cost_dir = train_dir + '/cost'
        summary_writer = tf.summary.FileWriter(cost_dir, s2s.sess.graph)
        data_train = tg.SequentialIterator(train_encoder_in, train_decoder_tar, batchsize=int(args.batch_size))
        data_valid = tg.SequentialIterator(valid_encoder_in, valid_decoder_tar, batchsize=500)

        steps = 0
        min_epoch_loss = float("inf")
        min_epoch = 0
        best_valid_path = train_dir
        best_epoch_path = train_dir
        best_t3_path = train_dir
        best_t1_path = train_dir
        best_cat_path = train_dir
        max_pred = {"max_pred_cat": [], "max_pred_t1": [], "max_pred_t3": []}
        max_prf = {"max_rf_cat": 0.0, "max_rp_cat": 0.0, "max_rr_cat": 0.0,
                   "max_rf_t1": 0.0, "max_rp_t1": 0.0, "max_rr_t1": 0.0,
                   "max_rf_t3": 0.0, "max_rp_t3": 0.0, "max_rr_t3": 0.0,
                   "max_epoch_cat": 0, "max_epoch_t1": 0, "max_epoch_t3": 0}

        for epoch in range(int(args.max_epoch)):
            print('epoch: ', epoch)
            print('..training')
            loss_epoch = []
            for news_batch, tweet_batch in data_train: # , lead_batch
                encoder_in, encoder_len, _ = batch(news_batch)
                decoder_target, decoder_len, de_mask = batch(tweet_batch, max_sequence_length=(max_twee_len + 3))
                # lead_target, lead_len = batch(lead_batch, max_twee_len+3)
                feed_dict = {encoder_input: encoder_in, en_in_len: encoder_len,
                             de_out: decoder_target, de_out_len: decoder_len,
                             decoder_mask: de_mask}  # lead: lead_target, lead_out_len: lead_len
                loss_value, train_op_value, summ = s2s.sess.run([loss, train_op, summ_op], feed_dict)

                loss_epoch.append(loss_value)
                summary_writer.add_summary(summ)
                steps += 1
                if steps % 100 == 0 and steps != 0:
                    summary_writer.flush()
                    predict = s2s.sess.run(decoder_prediction, feed_dict)
                    for i, (inp, pred) in enumerate(zip(feed_dict[de_out].T, predict.T)):
                        if i < 10:
                            print('sample {}'.format(i + 1))
                            inp_text = to_text(inp, vocab_inv)
                            print('input   > {}'.format(inp_text))
                            pred_text = to_text(pred, vocab_inv)
                            print('predict > {}'.format(pred_text))
                    max_prf, best_cat_path, best_t1_path, best_t3_path = run_inference(s2s, data_valid, encoder_input, en_in_len, decoder_prediction_inference,
                                            summary, news_con, vocab_inv, max_prf, max_pred, epoch, saver, train_dir, steps)

            print('epoch loss: {}'.format(sum(loss_epoch)))
            if sum(loss_epoch) < min_epoch_loss:
                min_epoch_loss = sum(loss_epoch)
                min_epoch = epoch
                """
                saver.save(s2s.sess, train_dir + '/best_epoch_checkpoint', global_step=epoch)
                print 'Saving Epoch model'
                best_epoch_path = train_dir + '/best_epoch_checkpoint-' + str(epoch)"""
            if (epoch - min_epoch) >= 2:
                break
        summary_writer.close()

        # print("*** The best model tested by batch_loss achieved at {} epoch and min_loss is {}! ***".
        #       format(min_epoch, min_epoch_loss))
        print("*** The best rouge results are ", max_prf)
    return best_epoch_path, best_t1_path, best_t3_path


def next_feed_inference(news, encoder_input, en_in_len):
    _encoder, _encoder_len, _ = batch(news)
    return {encoder_input: _encoder, en_in_len: _encoder_len}


def run_inference(s2s, data_valid, encoder_input, en_in_len, decoder_prediction, summary, news_con,
                  vocab_inv, max_prf, max_pred, epoch, saver, train_dir, steps):
    for news_batch, news_con_batch in data_valid:
        feed_dict = next_feed_inference(news_batch, encoder_input, en_in_len)
        predict = s2s.sess.run(decoder_prediction, feed_dict)
    predt = predict.T

    news_tfidf, pred_tfidf = [], []
    pnum = 0
    pred_eva = []
    pred_sig = []
    best_cat_path, best_t1_path, best_t3_path = train_dir, train_dir, train_dir
    for k in range(len(summary)):
        pred = []
        preds = []
        temp = []
        sumc = []
        for m in range(len(summary[k])):
            try:
                pred += predt[pnum + m].tolist()
                preds.append(predt[pnum + m].tolist())
                sumc += summary[k][m]
            except:
                import pdb; pdb.set_trace()
        pred_eva.append([pred])
        pred_sig.append(preds)
        temp.append(sumc)
        pnum += len(summary[k])
    extr_sum, reference = cosineSim(news_con, pred_eva, news_tfidf, pred_tfidf, vocab_inv, summary)
    print 'Concatenate summary:'
    result_cat = rouge(extr_sum, reference)
    extr_sum1, reference, extr_sum_top3 = cosineSimTop(news_con, pred_sig, news_tfidf, pred_tfidf,
                                                       vocab_inv, summary)
    print 'Extract 3 rouge:'
    result_t3 = rouge(extr_sum_top3, reference)
    print 'Extract 1 rouge:'
    result_t1 = rouge(extr_sum1, reference)
    if result_cat['ROUGE-1-F'] > max_prf['max_rf_cat']:
        max_prf['max_rf_cat'] = result_cat['ROUGE-1-F']
        max_prf['max_rp_cat'] = result_cat['ROUGE-1-P']
        max_prf['max_rr_cat'] = result_cat['ROUGE-1-R']
        max_prf['max_epoch_cat'] = epoch
        max_pred['max_pred_cat'] = predt.tolist()
        saver.save(s2s.sess, train_dir + '/best_cat_checkpoint', global_step=steps)
        print 'Saving Epoch model'
        best_cat_path = train_dir + '/best_cat_checkpoint-' + str(steps)
    if result_t1['ROUGE-1-F'] > max_prf['max_rf_t1']:
        max_prf['max_rf_t1'] = result_t1['ROUGE-1-F']
        max_prf['max_rp_t1'] = result_t1['ROUGE-1-P']
        max_prf['max_rr_t1'] = result_t1['ROUGE-1-R']
        max_prf['max_epoch_t1'] = epoch
        max_pred['max_pred_t1'] = predt.tolist()
        saver.save(s2s.sess, train_dir + '/best_t1_checkpoint', global_step=steps)
        print 'Saving Epoch model'
        best_t1_path = train_dir + '/best_t1_checkpoint-' + str(steps)
    if result_t3['ROUGE-1-F'] > max_prf['max_rf_t3']:
        max_prf['max_rf_t3'] = result_t3['ROUGE-1-F']
        max_prf['max_rp_t3'] = result_t3['ROUGE-1-P']
        max_prf['max_rr_t3'] = result_t3['ROUGE-1-R']
        max_prf['max_epoch_t3'] = epoch
        max_pred['max_pred_t3'] = predt.tolist()
        saver.save(s2s.sess, train_dir + '/best_t3_checkpoint', global_step=steps)
        print 'Saving Epoch model'
        best_t3_path = train_dir + '/best_t3_checkpoint-' + str(steps)
    print max_prf
    return max_prf, best_cat_path, best_t1_path, best_t3_path


def write_prediction(predict, summary, s2s, vocab_inv, train_dir, path_name, re):
    if re == True:
        path_name += 're'
    pred_dir = train_dir + '/best_' + path_name + '_predict/'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    predict = predict.tolist()
    tlt_num = 0
    for i in range(len(summary)):
        text_dir = os.path.join(pred_dir, str(i) + '.txt')
        text_s = 'Summary: \n'
        text_p = 'Prediction: \n'
        for j in range(len(summary[i])):
            text_s += s2s.to_text(summary[i][j], vocab_inv) + "\n"
            text_p += s2s.to_text(predict[tlt_num + j], vocab_inv) + "\n"
        tlt_num += len(summary[i])
        text = text_s + "\n" + text_p
        cPickle.dump(text, open(text_dir, 'wb'))


def evaluation_data(encoder_in, decoder_tar):
    data = list(zip(encoder_in, decoder_tar))
    shuffle(data)
    encoder_in, decoder_tar = zip(*data)
    train_num = int(math.ceil(len(encoder_in) * 0.8))
    train_encoder_in = encoder_in[:train_num]
    train_decoder_tar = decoder_tar[:train_num]
    eval_encoder_in = encoder_in[train_num:]
    eval_decoder_tar = decoder_tar[train_num:]
    return train_encoder_in, train_decoder_tar, eval_encoder_in, eval_decoder_tar


def reweight(news_con, news_grd_sim_rank, news_idx_clu, avg_weight_dict, lex_rank_score):
    alpha = float(args.alpha)
    add_news_id = {}
    add_news = {}
    top3 = 0
    tlt = 0
    base = 0
    for key in news_idx_clu.keys():
        news_idx = news_idx_clu[key]
        doc_id, clu_id = key.split(',')
        doc_id = int(doc_id)
        avg_weight = avg_weight_dict[key]
        avg_weight = avg_weight[0]
        clu_weight = []
        lex_rk = normalize(np.asarray(lex_rank_score[doc_id]).reshape(1, -1), norm='max')[0]
        for i in range(len(news_idx)):
            tweet_weight = avg_weight[i]
            lex_weight = lex_rk[news_idx[i]]
            fin_weight = alpha * tweet_weight + (1 - alpha) * lex_weight
            clu_weight.append(fin_weight)
        idx = [m[0] for m in sorted(enumerate(clu_weight), key=lambda x: x[1], reverse=True)]
        top_id_idx = idx[:3] if len(idx) > 3 else idx
        top_id = [x for i, x in enumerate(news_idx) if i in top_id_idx]

        add_news_id[key] = top_id
        news = []
        id_rk = []
        for id in top_id:
            news.append(news_con[doc_id][id])
            rk = min(news_grd_sim_rank[doc_id][id])
            id_rk.append(rk)
            if rk < 3:
                top3 += 1
        print "~~~ Top id {} rank {}".format(top_id, id_rk)
        tlt += len(top_id)
        base += 3
        add_news[key] = news
    print "== Precision for tweets vote top 3 news is ", (top3 * 1.0) / tlt
    print "== Recall for tweets vote top 3 news is ", (top3 * 1.0) / base
    return add_news_id, add_news


def re_train_data(news_grd_sim_rank, news_idx_clu, avg_weight_dict, news_con, data_info,
                  train_encoder_in, lex_rank_score):
    add_news_id, add_news = reweight(news_con, news_grd_sim_rank, news_idx_clu,
                                     avg_weight_dict, lex_rank_score)
    add_encoder_in = []
    add_decoder_tar = []
    for key in add_news:
        ele = {key: news_idx_clu[key]}
        try:
            idx = data_info.index(ele)
            news = add_news[key]
            en_in = [train_encoder_in[idx]] * len(news) * 5
            add_encoder_in += en_in
            for i in range(5):
                add_decoder_tar += news
        except:
            import pdb; pdb.set_trace()
    return add_encoder_in, add_decoder_tar


def lead_data(news_idx_clu, news_grd_sim_rank, data_info, news_con):
    top_rk = []
    data_lead = []
    for ele in data_info:
        key = ele.keys()[0]
        doc_id, clu_id = key.split(',')
        doc_id = int(doc_id)
        news_idx = news_idx_clu[key]
        news_rk = {}
        for id in news_idx:
            news_rk[id] = min(news_grd_sim_rank[doc_id][id])
        top_news = sorted(news_rk, key=news_rk.get)
        for e in top_news:
            if len(news_con[doc_id][e]) < 35:
                top_rk.append(news_rk[e])
                data_lead.append(news_con[doc_id][e])
                break
    print "Top rk for lead data: "
    print set(top_rk)
    import collections
    counter = collections.Counter(top_rk)
    print(counter)
    return data_lead


def main():
    summary, news, news_twee, tweets, vocab, vocab_inv, max_twee_len, news_con, title, first, \
        twe_num, news_num1, sum_org, news_org, tweet_org = loop_dir()
    vocab_size = len(vocab)
    print 'Vocabulary size is ', vocab_size
    top_news_grd, num3, rest, news_grd_sim_value, news_grd_sim_rank = grd_news(sum_org, news_org)
    pre_embed, _ = w2v1(vocab)
    """
    max_sen_rk = []
    i = 0
    for ele in news_grd_sim_rank:
        sen_rk = np.asarray(ele)
        sen_id = []
        for j in range(sen_rk.shape[1]):
            max_id = np.where(sen_rk[:, j] <= 2)
            sen_id += max_id[0].tolist()
        max_sen_rk.append(sen_id)
        print "Doc {} top 3 news locate at {} sentences, doc len is {}".format(i, sen_id, len(ele))
        i += 1
    """
    if args.mode == "cluster":
        print '== Loading data ...'
        data = cPickle.load(open('./epoch_tweet_news_Adam', 'rb'))
        news_idx_clu = data['news_idx_clu']
        avg_weight_dict = data['avg_weight_dict']
        all_data = cPickle.load(open('./all_data', 'rb'))
        train_encoder_in = all_data['train_encoder_in']
        train_decoder_tar = all_data['train_decoder_tar']
        decoder_infer_tar = all_data['decoder_infer_tar']
        decoder_infer_pin = all_data['decoder_infer_pin']
        data_info = all_data['data_info']
        clu_sen_len = all_data['clu_sen_len']
        train_dir = args.train_dir
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        print 'Loading data done! =='
    elif args.mode == "train":
        train_encoder_in, train_decoder_tar, decoder_infer_tar, decoder_infer_pin, \
            data_info, clu_sen_len, train_dir = train_data(args, vocab_size, pre_embed, news_con,
                                                           tweets, twe_num, summary, vocab_inv, news_grd_sim_rank)
    elif args.mode == "piece":
        train_encoder_in, train_decoder_tar, decoder_infer_tar, decoder_infer_pin, \
            data_info, clu_sen_len, train_dir = piece_data(args, vocab_size, pre_embed, summary,
                                                           news_con, tweets, twe_num, news_org,
                                                           sum_org, news_grd_sim_rank)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting seq2seq_attention in %s mode...', args.mode)

    tf.set_random_seed(1)
    s2s = unsuS2S_adv(vocab_size, args.embed_dim, args.batch_size, pre_embed, args, attention=True)
    """
    lex_rank_score = cPickle.load(open('./lex_rank_score', 'rb'))

    data_lead = lead_data(news_idx_clu, news_grd_sim_rank, data_info, news_con)
    print "Data lead length ", len(data_lead)
    
    add_encoder_in, add_decoder_tar = re_train_data(news_grd_sim_rank, news_idx_clu,
                                                    avg_weight_dict, news_con, data_info,
                                                    train_encoder_in, lex_rank_score)
    print 'length of add_encoder_in is ', len(add_encoder_in)
    print 'length of add_decoder_tar is ', len(add_decoder_tar)
    if args.all == True:
        add_encoder_in += train_encoder_in
        add_decoder_tar += train_decoder_tar
        print '== Using all the data retrain model'
        print '   Length of add_encoder_in is ', len(add_encoder_in)
        print '   Length of add_decoder_tar is ', len(add_decoder_tar)"""

    add_encoder_in, add_decoder_tar = train_encoder_in, train_decoder_tar

    max_de_len = max([len(ele) for ele in add_decoder_tar])
    """
    max_lead_len = max([len(ele) for ele in data_lead])
    max_len = max_de_len if max_de_len > max_lead_len else max_lead_len
    print "max_de_len: {},  max_lead_len: {},  max_len: {} ".format(max_de_len, max_lead_len, max_len)
    
    train_dir = train_dir + '/re'
    """
    data_lead = []
    best_epoch_path, best_t1_path, best_t3_path = run_training(s2s, add_encoder_in, add_decoder_tar, data_lead,
                                                               decoder_infer_tar, decoder_infer_pin, vocab_inv,
                                                               max_de_len, train_dir, summary, news_con)


if __name__ == '__main__':
    main()
