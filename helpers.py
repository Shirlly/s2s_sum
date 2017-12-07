import numpy as np


def word_encoder_batch(inputs, max_sen_len=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """
    sen_num = []
    sen_len = []
    for i in range(len(inputs)):
        sen_num.append(len(inputs[i]))
        sen_len.append([len(s) for s in inputs[i]])

    max_sen_num = max(sen_num)
    word_batch_size = len(inputs) * max_sen_num   # [clu_num * max_sen_num]

    pad_sen_len = []
    for i in range(len(inputs)):
        temp_sen_pad = np.zeros(shape=[max_sen_num], dtype=np.int32)
        real_sen_len = np.asarray(sen_len[i])
        for j in range(len(real_sen_len)):
            temp_sen_pad[j] = temp_sen_pad[j] + real_sen_len[j]
        pad_sen_len += temp_sen_pad.tolist()

    if max_sen_len is None:
        max_sen_len = max(pad_sen_len)

    inputs_batch_major = np.zeros(shape=[word_batch_size, max_sen_len], dtype=np.int32)
    sen_padding_mask = np.zeros(shape=[len(inputs), max_sen_num], dtype=np.int32)

    for i, seq in enumerate(inputs):
        for j in range(sen_num[i]):
            sen_padding_mask[i, j] = 1
            row = max_sen_num * i + j
            for k, ele in enumerate(inputs[i][j]):
                inputs_batch_major[row, k] = ele

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return {'inputs_time_major': inputs_time_major, 'sen_padding_mask': sen_padding_mask,
            'sen_num': sen_num, 'sen_len': pad_sen_len}


def batch(inputs, re=False, max_sequence_length=None):  # None
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    batch_size = len(inputs)
    sequence_lengths = [len(seq) for seq in inputs]
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    """
    else:
        sequence_lengths = [max_sequence_length] * batch_size
    """

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
    decoder_mask = np.zeros(shape=[batch_size, max_sequence_length+1], dtype=bool)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            try:
                inputs_batch_major[i, j] = element
                decoder_mask[i, j] = True
            except:
                import pdb; pdb.set_trace()
    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    decoder_mask = decoder_mask.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths, decoder_mask


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
        raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]