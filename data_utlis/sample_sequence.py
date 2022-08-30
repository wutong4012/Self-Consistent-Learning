import torch
import math
import torch.nn.functional as F


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        # convert to 1D
        #logits = logits.view(logits.size()[1]).contiguous()
        #logits = logits.contiguous()
        sorted_logits, sorted_indices = torch.sort(
            logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for i in range(sorted_indices.size()[0]):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
        #indices_to_remove = sorted_indices[sorted_indices_to_remove]
        #logits[indices_to_remove] = filter_value
        # going back to 2D
        #logits = logits.view(1, -1).contiguous()
    return logits


def enforce_repetition_penalty(lprobs, prev_output_tokens, repetition_penalty=1.5):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for previous_token in set(prev_output_tokens):
        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
        if lprobs[previous_token] < 0:
            lprobs[previous_token] *= repetition_penalty
        else:
            lprobs[previous_token] /= repetition_penalty


def switch(next_value, init, is_update):  # 换成真实token
    is_update = is_update.type_as(next_value)
    return (1-is_update)*init + is_update*next_value


def get_atten_mask(batch_size, seq_length, memory_length=0):
    memory_attention_mask = torch.ones(
        (batch_size, 1, seq_length, seq_length + memory_length), dtype=torch.long)
    memory_attention_mask = torch.tril(
        torch.triu(memory_attention_mask, 1 - seq_length + memory_length), memory_length)

    return memory_attention_mask  # [bs, 1, seq_len, seq_len+M]


def sample_sequence_batch(model, context_tokens_tensor, context_length_tensor, max_out_seq=None, mems=None,
                          end_token_id=None, repetition_penalty=1.0, temperature=1.0, top_k=0, top_p=0.0):
    """_summary_

    Args:
        model (_type_): _description_
        context_tokens_tensor (Tensor): [bs, seq_len]
        context_length_tensor (Tensor): [bs, ]
        max_out_seq (_type_, optional): _description_. Defaults to None.
        mems (_type_, optional): _description_. Defaults to None.
        end_token_id (_type_, optional): _description_. Defaults to None.
        repetition_penalty (float, optional): _description_. Defaults to 1.0.
        temperature (float, optional): _description_. Defaults to 1.0.
        top_k (int, optional): _description_. Defaults to 0.
        top_p (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    org_context_length = torch.min(context_length_tensor).item()
    batch_size = context_tokens_tensor.shape[0]
    tokens = context_tokens_tensor[:, :org_context_length]
    attention_mask = get_atten_mask(batch_size, org_context_length).cuda()

    counter, mem_length = 0, 0
    if mems is None:
        mems = []
    if end_token_id is None:
        end_token_id = 50000
    if max_out_seq is None:
        max_out_seq = 512
    log_probs_tensor = torch.tensor([0.0] * batch_size)
    count_num = torch.tensor([0] * batch_size)
    output_tokens_lists, log_probs_list = [], []
    with torch.no_grad():
        while counter < max_out_seq:
            index = org_context_length + counter
            if counter == 0:
                logits, *mems = model.forward(input_ids=tokens, position_ids=None, 
                                              attention_mask=attention_mask, mems=mems)
            else:
                logits, *mems = model.forward(input_ids=tokens[:, index - 1: index], position_ids=None, 
                                              attention_mask=tokens.new_ones(batch_size, 1, 1, mem_length + 1), mems=mems)
            logits = logits[:, -1]
            logits /= temperature
            logits = top_k_logits(logits, top_k=top_k, top_p=top_p)
            if repetition_penalty != 1.0:
                for bz in range(batch_size):
                    enforce_repetition_penalty(logits[bz, :], tokens[bz, :], repetition_penalty)
            probs = F.softmax(logits, dim=-1)  # [bs, vocab_size]
            
            prev = torch.multinomial(probs, num_samples=1).view(-1)  # [bs]

            if index < torch.max(context_length_tensor).item():
                prev = switch(
                    prev, context_tokens_tensor[:, index], context_length_tensor <= index)
            for i in range(batch_size):
                if index > context_length_tensor[i] and prev[i] != end_token_id and probs[i][prev[i]] != 0:
                    log_probs_tensor[i] += math.log(probs[i][prev[i]])
                    # log_probs_tensor[i] += probs[i][prev[i]].item()
                    count_num[i] += 1
                if prev[i] == end_token_id:
                    # log_probs_tensor[i] /= (-count_num[i])
                    log_probs_tensor[i] /= count_num[i]

            if torch.all(prev == end_token_id).item():
                break

            finished = tokens[prev == end_token_id]
            output_tokens_lists.extend(finished.detach().cpu().tolist())
            log_probs_list.extend(log_probs_tensor[prev == end_token_id])

            # continue with non-ending tokens
            conti_idx = (prev != end_token_id)
            tokens, prev = tokens[conti_idx], prev[conti_idx]
            context_tokens_tensor = context_tokens_tensor[conti_idx]
            context_length_tensor = context_length_tensor[conti_idx]
            
            log_probs_tensor = log_probs_tensor[conti_idx]
            count_num = count_num[conti_idx]
            
            batch_size = tokens.shape[0]
            for im in range(len(mems)):
                mems[im] = mems[im][conti_idx, :, :]

            tokens = torch.cat((tokens, prev.view(batch_size, 1)), dim=-1)

            counter += 1

    output_tokens_lists.extend(tokens.detach().cpu().tolist())
    log_probs_list.extend(log_probs_tensor)
    output_tokens_lists = [tokens[:tokens.index(
        end_token_id)] if end_token_id in tokens else tokens for tokens in output_tokens_lists]
    ppl_list = [math.exp(i) for i in log_probs_list]
    # log_probs_list = [i.item() for i in log_probs_list]
    return {
        'ids_list': output_tokens_lists,
        'ppl_list': ppl_list,
        # 'prob_list': log_probs_list,
    }


def sample_sequence(model, tokens, attention_mask, do_sampling=True,
                    repetition_penalty=1.0, max_out_seq=None, mems=None, end_token_id=None,
                    mem_length=0, temperature=1.0, top_k=0, top_p=0.0):
    """_summary_

    Args:
        model (_type_): _description_
        tokens (Tensor): [1, seq_len]
        attention_mask (Tensor): [1, 1, seq_len, seq_len]
        do_sampling (bool, optional): _description_. Defaults to True.
        repetition_penalty (float, optional): _description_. Defaults to 1.0.
        max_out_seq (_type_, optional): _description_. Defaults to None.
        mems (_type_, optional): _description_. Defaults to None.
        end_token (_type_, optional): _description_. Defaults to None.
        mem_length (int, optional): _description_. Defaults to 0.
        temperature (float, optional): _description_. Defaults to 1.0.
        top_k (int, optional): _description_. Defaults to 0.
        top_p (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    counter = 0
    if mems is None:
        mems = []
    if end_token_id is None:
        end_token_id = 50000
    if max_out_seq is None:
        max_out_seq = 512
    org_context_length = tokens.size(1)
    with torch.no_grad():
        # while counter < (max_out_seq - org_context_length):
        while counter < max_out_seq:
            if counter == 0:
                logits, *mems = model(input_ids=tokens, position_ids=None,
                                      attention_mask=attention_mask, mems=mems)
            else:
                index = org_context_length + counter
                logits, *mems = model(input_ids=tokens[:, index - 1: index], position_ids=None,
                                      attention_mask=tokens.new_ones(1, 1, 1, mem_length + 1), mems=mems)
            logits = logits[:, -1]
            logits /= temperature
            if do_sampling:
                logits = top_k_logits(logits, top_k=top_k, top_p=top_p)
            log_probs = F.softmax(logits, dim=-1)

            if repetition_penalty != 1.0:
                enforce_repetition_penalty(
                    log_probs[0, :], tokens[0, :], repetition_penalty)
            prev = torch.multinomial(log_probs, num_samples=1)[0]
            is_end = (prev == end_token_id)
            if is_end:
                break
            tokens = torch.cat((tokens, prev.view(1, 1)), dim=1)
            counter += 1

    output_tokens_list = tokens.detach().cpu().tolist()
    if end_token_id in output_tokens_list:
        output_tokens_list = output_tokens_list[:output_tokens_list.index(
            end_token_id)]

    return output_tokens_list[0], mems
