import os
import util
import time
import math
import argparse
import helpers
import cPickle
import numpy as np
import tensorflow as tf
import tensorgraph as tg
from random import shuffle
from data_add import loop_dir
from helpers import word_encoder_batch, batch
from unsupervisedS2Sadv2 import unsuS2S_adv
from preprare_cluster_data import assignTweet
from extractSummary import rouge, cosineSim, cosineSimTop
from cluster_process import cluster_demo, grd_news

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
parser.add_argument("--max_grad_norm", default=2.0, help='for gradient clipping')
parser.add_argument("--adagrad_init_acc", default=0.1, help='initial accumulator value for Adagrad')
parser.add_argument("--opt", default='Adagrad', help='Optimization method, Adam or Adagrad')
parser.add_argument("--valid", default=True, help='Adopt valid training mechanism')
args = parser.parse_args()

PAD = 0


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
    _encoder, _encoder_len = helpers.batch(news)
    return {encoder_input: _encoder, en_in_len: _encoder_len}


def cluster_data(vocab_size, pre_embed, news_con, tweets, twe_num, summary, vocab_inv):
    news2, news_num = conv3to2(news_con)
    tweet3 = conv2to3(tweets, twe_num, news_con)
    s2s = unsuS2S_adv(vocab_size, args.embed_dim, args.batch_size, pre_embed, attention=True)

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

    if len(news_num) == len(twe_num) == len(summary):
        cluster = cluster_demo(news3, news_num, summary)
        print 'Clustering Done'

        encoder_in, decoder_tar, decoder_infer_tar, decoder_infer_pin, data_info, clu_sen_len \
            = assignTweet(cluster, news3, twee3, tweet3, news_con, news_num, summary, vocab_inv)

        return encoder_in, decoder_tar, decoder_infer_tar, decoder_infer_pin, data_info, clu_sen_len
    else:
        raise Exception('Number not matching !!')


def run_training(s2s, train_encoder_in, train_decoder_tar, valid_encoder_in, valid_decoder_tar, vocab_inv,
                 max_twee_len):
    with s2s.graph.as_default():
        encoder_input, en_in_len, encoder_output, encoder_final_state = s2s.encoder()
        de_out, de_out_len, title_out, first_out, decoder_logits, decoder_prediction, loss_weight, \
        decoder_target, decoder_title, decoder_first, decoder_prediction_inference \
            , attention_values, attention_keys, decoder_state_train, decoder_context_state_train \
            = s2s.decoder_adv(max_twee_len)

        train_dir = args.log_root + '_' + args.opt + '_plain'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        s2s.build_graph()
        saver = tf.train.Saver()
        if args.model_restore:
            saver.restore(s2s.sess, './log2_gru_embed21/pre_check_point-49')
        else:
            tf.set_random_seed(1)
            init = tf.global_variables_initializer()
            s2s.sess.run(init)
        summary_writer = tf.summary.FileWriter(train_dir + '/cost', s2s.sess.graph)
        data_train = tg.SequentialIterator(train_encoder_in, train_decoder_tar, batchsize=int(args.batch_size))
        if args.valid:
            data_valid = tg.SequentialIterator(valid_encoder_in, valid_decoder_tar, batchsize=int(args.batch_size))

        steps = 0
        min_valid_loss = float("inf")
        optimal_step = 0
        opt = False
        min_epoch_loss = float("inf")
        min_epoch = 0
        best_valid_path = train_dir
        best_epoch_path = train_dir
        for epoch in range(int(args.max_epoch)):
            print('epoch: ', epoch)
            print('..training')
            loss_epoch = []
            for news_batch, tweet_batch in data_train:
                encoder_input, encoder_len = batch(news_batch)
                decoder_target, decoder_len = batch(tweet_batch)  # , max_twee_len
                loss, train_op, summ, global_step = s2s.run_train_step(encoder_input, encoder_len,
                                                                       decoder_target, decoder_len)
                loss_epoch.append(loss)
                summary_writer.add_summary(summ, global_step)
                steps += 1
                if steps % 100 == 0 and steps != 0:
                    summary_writer.flush()
                    s2s.run_train_result(encoder_input, encoder_len, decoder_target, decoder_len, vocab_inv)
                    valid_loss = []
                    print 'Step {} for validation '.format(steps)
                    if args.valid:
                        for news_valid, tweet_valid in data_valid:
                            encoder_input, encoder_len = batch(news_valid)
                            decoder_target, decoder_len = batch(tweet_valid, max_twee_len)
                            valid_loss.append(s2s.run_valid_step(encoder_input, encoder_len, decoder_target, decoder_len))
                        s2s.run_valid_result(encoder_input, encoder_len, decoder_target, decoder_len, vocab_inv)
                        if sum(valid_loss) < min_valid_loss:
                            min_valid_loss = sum(valid_loss)
                            optimal_step = steps
                            saver.save(s2s.sess, train_dir + '/best_valid_checkpoint', global_step=global_step)
                            print 'Saving Valid model'
                            best_valid_path = train_dir + '/best_valid_checkpoint-' + str(global_step)
                        if (steps - optimal_step) % 100 > 10:
                            opt = True
                            break
                # if opt:
                #     break
            if opt:
                break
            print('epoch loss: {}'.format(sum(loss_epoch)))
            if loss_epoch < min_epoch_loss:
                min_epoch_loss = loss_epoch
                min_epoch = epoch
                saver.save(s2s.sess, train_dir + '/best_epoch_checkpoint', global_step=epoch)
                print 'Saving Epoch model'
                best_epoch_path = train_dir + '/best_epoch_checkpoint-' + str(epoch)
        summary_writer.close()

        if args.valid:
            print("Best running step is ", optimal_step)
            print("Minimum validation loss is ", min_valid_loss)
            print("*** Running end after {} epochs and {} iterations!!! ***".format(epoch, steps))
        print("*** The best model tested by batch_loss achieved at {} epoch ! ***".format(min_epoch))
    return best_valid_path, best_epoch_path, train_dir


def calculate_news_weight(s2s, encoder_in, decoder_tar, data_info, clu_sen_len, news_grd_sim_rank, max_twee_len,
                          best_valid_path, best_epoch_path, train_dir):
    with s2s.graph.as_default():
        s2s.build_graph()
        saver = tf.train.Saver()

        def weight_calculation(path):
            saver.restore(s2s.sess, path)
            data_tlt = tg.SequentialIterator(encoder_in, decoder_tar, batchsize=128)
            attn_weight_tlt = []
            for news_batch, tweet_batch in data_tlt:
                encoder_input, encoder_len = batch(news_batch)
                decoder_target, decoder_len = batch(tweet_batch)  # , max_twee_len
                attn_weight = s2s.run_attn_weight(encoder_input, encoder_len, decoder_target, decoder_len)
                attn_weight_tlt += attn_weight.tolist()

            attn = []
            if len(clu_sen_len) == len(attn_weight_tlt):
                for i in range(len(encoder_in)):
                    sen_attn = []
                    for j in range(len(clu_sen_len[i])):
                        snum = sum(clu_sen_len[i][:j])
                        enum = snum + clu_sen_len[i][j]
                        sen_attn.append(sum(attn_weight_tlt[i][snum:enum]))
                    attn.append(sen_attn)

            weight_dict = {}
            news_idx_clu = {}
            try:
                assert len(data_info) == len(attn)
            except:
                import pdb;
                pdb.set_trace()
            for i in range(len(attn)):
                key = data_info[i].keys()[0]
                if key in weight_dict:
                    value = weight_dict[key]
                    value.append(attn[i])
                    weight_dict[key] = value
                else:
                    value = []
                    weight_dict[key] = value.append(attn[i])
                    news_idx_clu[key] = data_info[i][key]

            top3 = 0
            base = 0
            tlt = 0
            avg_weight_dict = {}
            twee_top_news = {}
            for key in weight_dict.keys():
                value = np.asarray(weight_dict[key])
                clu_twee_num, clu_news_num = value.shape
                news_idx = news_idx_clu[key]
                if clu_twee_num > 0:
                    clu_avg_weight = np.divide(np.sum(value, axis=0), clu_twee_num * 1.0)
                    avg_weight_dict[key] = clu_avg_weight
                    news_id = [m[0] for m in sorted(enumerate(clu_avg_weight), key=lambda x: x[1], reverse=True)]
                    top_id_idx = news_id[:3] if len(news_id) > 3 else news_id
                    top_id = [x for i, x in enumerate(news_idx) if i in top_id_idx]
                    twee_top_news[key] = top_id
                    doc_id, _ = key.split(",")
                    for ele in top_id:
                        rk = min(news_grd_sim_rank[doc_id][ele])
                        if rk < 3:
                            top3 += 1
                    tlt += len(top_id)
                    base += 3
                else:
                    twee_top_news[key] = []

            print "== Precision for tweets vote top 3 news is ", (top3 * 1.0) / tlt
            print "== Recall for tweets vote top 3 news is ", (top3 * 1.0) / base
            return twee_top_news

        print "*** Weight calculation based on best valid result ***"
        weight_calculation(best_valid_path)
        print "*** Weight calculation based on best epoch result ***"
        weight_calculation(best_epoch_path)


def run_inference(s2s, decoder_infer_tar, decoder_infer_pin, summary, news_con, vocab_inv,
                  best_valid_path, best_epoch_path, train_dir):
    with s2s.graph.as_default():
        s2s.build_graph()
        saver = tf.train.Saver()
        data_infer = tg.SequentialIterator(decoder_infer_tar, decoder_infer_pin, batchsize=500)

        def prediction(path, name):
            saver.restore(s2s.sess, path)
            for news_batch, newsc in data_infer:
                encoder_input, encoder_len = batch(news_batch)
                predict = s2s.run_inference(encoder_input, encoder_len)
                write_prediction(predict, summary, s2s, vocab_inv, train_dir, name)

        print "==== Restore model from best valid path"
        prediction(best_valid_path, "valid")
        print "==== Restore model from best epoch path"
        prediction(best_epoch_path, "epoch")


def write_prediction(predict, summary, s2s, vocab_inv, train_dir, path_name):
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


def main():
    summary, news, news_twee, tweets, vocab, vocab_inv, max_twee_len, news_con, title, first, \
        twe_num, news_num1, sum_org, news_org, tweet_org = loop_dir()
    vocab_size = len(vocab)
    print 'Vocabulary size is ', vocab_size
    max_twee_len = max_twee_len + 3
    top_news_grd, num3, rest, news_grd_sim_value, news_grd_sim_rank = grd_news(sum_org, news_org)
    pre_embed = []

    encoder_in, decoder_tar, decoder_infer_tar, decoder_infer_pin, data_info, clu_sen_len = \
        cluster_data(vocab_size, pre_embed, news_con, tweets, twe_num, summary, vocab_inv)

    train_encoder_in, train_decoder_tar, valid_encoder_in, valid_decoder_tar = encoder_in, decoder_tar, [], []
    if args.valid:
        train_encoder_in, train_decoder_tar, valid_encoder_in, valid_decoder_tar = \
            evaluation_data(encoder_in, decoder_tar)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting seq2seq_attention in %s mode...', args.mode)

    tf.set_random_seed(1)
    s2s = unsuS2S_adv(vocab_size, args.embed_dim, args.batch_size, pre_embed, attention=True)
    best_valid_path, best_epoch_path, train_dir = '', '', ''
    if args.mode == 'train':
        best_valid_path, best_epoch_path, train_dir = run_training(s2s, train_encoder_in, train_decoder_tar,
                                                                   valid_encoder_in, valid_decoder_tar,
                                                                   vocab_inv, max_twee_len)
        run_inference(s2s, decoder_infer_tar, decoder_infer_pin, summary, news_con, vocab_inv,
                      best_valid_path, best_epoch_path, train_dir)
        # calculate_news_weight(s2s, encoder_in, decoder_tar, data_info, clu_sen_len, news_grd_sim_rank, max_twee_len,
        #                       best_valid_path, best_epoch_path, train_dir)
    elif args.mode == 'inference':
        run_inference(s2s, decoder_infer_tar, decoder_infer_pin, summary, news_con, vocab_inv,
                      best_valid_path, best_epoch_path, train_dir)
        calculate_news_weight(s2s, encoder_in, decoder_tar, data_info, clu_sen_len, news_grd_sim_rank, max_twee_len,
                              best_valid_path, best_epoch_path, train_dir)


if __name__ == '__main__':
    main()
