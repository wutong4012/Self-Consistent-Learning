import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')


# slight shuffle such that |sigma[i]-i| <= k
def word_shuffle(x, k, bos_token, pad_token):
    base = torch.arange(x.size(1), dtype=torch.float).repeat(x.size(0), 1)
    inc = (k+1) * torch.rand(x.size())
    inc[x == bos_token] = 0     # do not shuffle the start sentence symbol
    inc[x == pad_token] = k+1  # do not shuffle end paddings
    _, sigma = (base + inc).sort(dim=1)
    return x[torch.arange(x.size(0)).unsqueeze(0).t(), sigma]


def word_drop(x, p, pad_token):     # drop words with probability p
    x_ = []
    for i in range(x.size(0)):
        words = x[i, :].tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = True  # do not drop the start sentence symbol
        sent = [w for j, w in enumerate(words) if keep[j]]
        sent += [pad_token] * (len(words)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).contiguous().to(x.device)


# substitute words with probability p
def word_substitute(x, p, bos_token, pad_token, vocab_size):
    rand_idx = (torch.rand(x.size(), device=x.device) < p) & (
        x != pad_token) & (x != bos_token)
    random_words = torch.randint(
        vocab_size, x.shape, dtype=torch.long).to(x.device)
    x[rand_idx] = random_words[rand_idx]
    return x


def noisy(x, drop_prob, sub_prob, shuffle_dist, bos_token, pad_token, vocab_size):
    if shuffle_dist > 0:
        x = word_shuffle(x, shuffle_dist, bos_token, pad_token)
    if drop_prob > 0:
        x = word_drop(x, drop_prob, pad_token)
    if sub_prob > 0:
        x = word_substitute(x, sub_prob, bos_token, pad_token, vocab_size)
    return x
