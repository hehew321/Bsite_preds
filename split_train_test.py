import pandas as pd
import os

# , 'C17ORF85', 'C22ORF28'
for rbp_name in ['ALKBH5', 'C22ORF28', 'C17ORF85', 'FXR1', 'TAF15', 'QKI', 'WTAP']:
    f = open("./get_data/fasta/{}.fasta".format(rbp_name))
    content = f.readlines().copy()
    length = len(content)
    if not os.path.isdir("./data/{}".format(rbp_name)):
        os.system("mkdir ./data/{}".format(rbp_name))
    f_train = open("./data/{0}/{0}_train.fasta".format(rbp_name), 'w')
    f_test = open("./data/{0}/{0}_test.fasta".format(rbp_name), 'w')

    end = int(length*0.8)
    if content[end][0] != ">":
        end += 1
    print("length:{}, end: {}".format(length, end))
    while content[end][:17] == content[end-2][:17]:
        end += 2
    print(end)
    f_train.writelines(content[:end])
    f_test.writelines(content[end:])
    f.close()
    f_test.close()
    f_train.close()
