import os
import tqdm
import pandas as pd

# 'FXR1','QKI', 'TAF15', 'WTAP'
for rbp_csv in os.listdir('./circRNA_dataset/bindingsites'):
# for rbp in []:
#     rbp_csv = rbp+'.csv'
    rna_name, start_site, end_site = [], [], []
    f_sites = open("./circRNA_dataset/bindingsites/{}".format(rbp_csv))
    f_seq = open("./circRNA_dataset/human_hg19_circRNAs.fa")
    i = 0
    for line in f_sites.readlines():
        i += 1
        if i == 1:
            continue
        _line = line.strip().split()
        rna_name.append(_line[0])
        start_site.append(_line[-2])
        end_site.append(_line[-1])
    all_rna_name, all_seq = [], []
    # print(len(rna_name), len(start_site), len(end_site))
    for line in f_seq.readlines():
        if line[0] == ">":
            all_rna_name.append(line[1:17])
        else:
            all_seq.append(line.strip())
    assert len(all_rna_name) == len(all_seq)

    seq = []
    for seq_name in rna_name:
        for i, name in enumerate(all_rna_name):
            if name == seq_name:
                seq.append(all_seq[i].strip())
                break
    assert len(rna_name) == len(seq)
    # print(len(rna_name), len(start_site), len(end_site), len(end_site))
    save_pd = pd.DataFrame({"CircRNA Tag Name": rna_name[:10000], "Start Site": start_site[:10000], "End Site": end_site[:10000]})
    save_pd["Seq"] = seq[:10000]
    save_pd.to_csv('./data_csv/{}'.format(rbp_csv), index=False)
    f_sites.close()
    f_seq.close()
    print("Completed the generation of {}".format(rbp_csv[:-4]))



