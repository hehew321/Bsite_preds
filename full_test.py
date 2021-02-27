from keras.models import load_model
from old_data import full_test_data
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


name = "TAF15"
model = load_model("./models/{}.h5".format(name))
seq, label, len_seq, rbp_name = full_test_data(window_size=21, rbp_name=name)
ouput = model.predict(seq).reshape(-1)
for i in range(len(len_seq)):
    plt.plot([i for i in range(len_seq[i])], ouput[sum(len_seq[:i]): sum(len_seq[:i+1])])
    plt.scatter([i for i in range(len_seq[i])], label[sum(len_seq[:i]): sum(len_seq[:i+1])], c='r', marker='*')
    plt.title(rbp_name[i])
    plt.show()
