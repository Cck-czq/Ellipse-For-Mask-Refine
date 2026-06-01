import numpy as np
logits_list=np.load("../logits_list/UWCOT220_000012.npy")
for logit_array in logits_list:
    print(logit_array.shape)