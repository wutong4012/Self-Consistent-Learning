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


def embd_noise(embds, noise_type='hollow', zeta='0.0'):
    embeds_magn = torch.norm(embds, dim=-1)  # LB
    noise = torch.rand_like(embds)  # LBE
    if(noise_type == 'hollow'):
        noise = (zeta * embeds_magn).unsqueeze(-1) * \
            torch.nn.functional.normalize(
                noise, p=2.0, eps=1e-12, dim=-1)  # LBE
    elif(noise_type == 'centered-gau'):  # gaussian hypersphere, mu = 0, var = (zeta/3)^2
        noise_magn = zeta/3 * \
            torch.randn(embeds_magn.size(0), embeds_magn.size(1),
                        device=embds.device)  # LB
        noise = (noise_magn * embeds_magn).unsqueeze(-1) * \
            torch.nn.functional.normalize(
                noise, p=2.0, eps=1e-12, dim=-1)  # LBE
    elif(noise_type == 'uniform'):  # uniform hypersphere in [0, zeta]
        noise_magn = zeta * \
            torch.rand(embeds_magn.size(0), embeds_magn.size(
                1), device=embds.device)  # LB
        noise = (noise_magn * embeds_magn).unsqueeze(-1) * \
            torch.nn.functional.normalize(
                noise, p=2.0, eps=1e-12, dim=-1)  # LBE
    elif(noise_type == 'shifted-gau'):  # gaussian hypersphere, mu = zeta, var = 1
        noise_magn = zeta + \
            torch.randn(embeds_magn.size(0), embeds_magn.size(1),
                        device=embds.device)  # LB
        noise = (noise_magn * embeds_magn).unsqueeze(-1) * \
            torch.nn.functional.normalize(
                noise, p=2.0, eps=1e-12, dim=-1)  # LBE
    else:
        exit("args.noise_type is not a valid option")
    return noise


def mask_tokens(inputs, tokenizer, mlm_prob=0.15):
    ''' Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. '''
    # We sample a few tokens in each sequence for masked-LM training 
    # (with probability config.model_params.mlm_prob defaults to 0.15 in Bert/RoBERTa)

    masked_indices = torch.bernoulli(
        torch.full(inputs.shape, mlm_prob)).to(torch.uint8)

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8))
    indices_replaced[:, 0], indices_replaced[:, -1] = 0, 0  # [CLS] and [SEP] don't change
    indices_replaced = indices_replaced.to(torch.uint8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5))
    random_words = torch.randint(
        len(tokenizer), inputs.shape, dtype=torch.long).to(inputs.device)
    indices_random[:, 0], indices_random[:, -1] = 0, 0
    indices_random = indices_random.to(torch.uint8) & masked_indices & ~indices_replaced
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs