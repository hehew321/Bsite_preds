import sys
import numpy as np


def get_data(window_size, rbp_name, train=True):
    """
    get data and label
    """
    f_name_seq = open("/data/wuhehe/wuhe/Bsite_data/{0}/{0}_seq.fasta".format(rbp_name))
    all_name_seq = {}
    line_list = f_name_seq.readlines()

    for i, line in enumerate(line_list):
        _line = line.strip().split()
        if _line[0][0] == '>':
            all_name_seq[_line[0][1:]] = line_list[i+1].strip()
    f_name_seq.close()
    # print(all_name_seq)
    if train:
        f_p = open("/data/wuhehe/wuhe/Bsite_data/{0}/{0}_train.fasta".format(rbp_name))
    else:
        f_p = open("/data/wuhehe/wuhe/Bsite_data/{0}/{0}_test.fasta".format(rbp_name))

    name_y, name_seq = {}, {}
    lines_copy = f_p.readlines()

    for line in lines_copy:
        _line = line.strip().split()
        if _line[0][0] == ">":
            _name = _line[0][1:]
            _seq = all_name_seq[_name]
            name_y[_name] = np.zeros(len(_seq))
            name_seq[_name] = _seq

    for i in range(len(lines_copy)):
        line = lines_copy[i].strip().split()
        name_y[line[0][1:17]][int(line[-2])-1:int(line[-1])] = 1.

    rna_name = list(name_y.keys())
    assert len(name_y) == len(name_seq)
    data_pos, data_neg = [], []
    for i in range(len(rna_name)):
        _seq = name_seq[rna_name[i]]
        _y = name_y[rna_name[i]]

        for j in range(len(_seq)-window_size//2):
            if _y[j+window_size//2] == 1:
                data_pos.append(_seq[j: j+window_size])
            else:
                if len(data_neg) > 2000000:
                    # break
                    continue
                data_neg.append(_seq[j: j+window_size])

    data_pos = to_matrix(data_pos, window_size=window_size)
    data_neg = to_matrix(data_neg, window_size=window_size)
    f_p.close()
    return data_pos, data_neg


def full_test_data(window_size, rbp_name):
    f_p = open("/data/wuhehe/wuhe/Bsite_preds/data/{0}/{0}_test_full.fasta".format(rbp_name))
    rna_name, seq = [], []
    name_y, name_seq = {}, {}
    lines_copy = f_p.readlines()

    for line in lines_copy:
        _line = line.strip().split()
        if _line[0][0] == ">":
            rna_name.append(_line[0][1:])
        else:
            seq.append(_line)
            name_y[rna_name[-1]] = np.zeros(len(_line[0]))
            name_seq[rna_name[-1]] = _line[0]

    for i in range(0, len(lines_copy)):
        line = lines_copy[i].strip().split()
        name_y[line[0][1:17]][int(line[-2]):int(line[-1])] = 1.

    rna_name = list(set(rna_name))
    assert len(rna_name) == len(name_y) == len(name_seq)
    _data = []
    _y = []
    len_seq = []
    name = []
    for i in range(len(rna_name)):
        _seq = name_seq[rna_name[i]]
        name.append(rna_name[i])
        _y.extend(name_y[rna_name[i]])
        len_seq.append(len(_seq))

        for j in range(len(_seq)):
            if j < window_size//2:
                _data.append((window_size-len(_seq[:j+window_size//2])-1)*'0'+_seq[:j+window_size//2+1])
            elif j >= len(_seq)-window_size//2:
                _data.append(_seq[j-window_size//2:]+'0'*(window_size-len(_seq[j-window_size//2:])))
            else:
                _data.append(_seq[j-window_size//2: j+window_size//2+1])
            # print(_data[-1], len(_data[-1]))

    _data = to_matrix(_data, window_size=window_size)
    return _data, _y, len_seq, name


def to_matrix(seq, window_size):
    seq_data = []
    for i in range(len(seq)):
        mat = np.array([0.] * 4 * window_size).reshape(window_size, 4)
        for j in range(len(seq[i])):
            # print(seq[i])
            if seq[i][j] == 'A':
                mat[j][0] = 1.0
            elif seq[i][j] == 'C':
                mat[j][1] = 1.0
            elif seq[i][j] == 'G':
                mat[j][2] = 1.0
            elif seq[i][j] == 'U' or seq[i][j] == 'T':
                mat[j][3] = 1.0
            elif seq[i][j] == 'N':
                mat[j] = 0.25
            elif seq[i][j] == '0':
                mat[j] = 0
            else:
                print("Presence of unknown nucleotides")
                sys.exit()
        seq_data.append(mat)
    return np.array(seq_data)


if __name__ == "__main__":
    _window_size, _rbp_name = 21, "AGO1"
    _data_pos, _data_neg = get_data(window_size=_window_size, rbp_name=_rbp_name)
    print(_data_pos)
    print(_data_pos.shape, _data_neg.shape)
    # seq, label, len_seq, name = get_data(window_size=21, rbp_name='AGO1', train=True)
    # print(seq.shape)
    # print(len(label))
    # print(len_seq)




