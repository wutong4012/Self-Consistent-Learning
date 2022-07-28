import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from data_utlis.noisy_input_ids import mask_tokens, noisy


def get_atten_mask(max_length, memory_length=0):
    memory_attention_mask = torch.ones(
        (max_length, max_length + memory_length), dtype=torch.long)
    memory_attention_mask = torch.tril(
        torch.triu(memory_attention_mask, 1 - max_length + memory_length), memory_length)

    return memory_attention_mask  # [seq_len, seq_len+M]


def padding_memory_mask(mam: list, max_length):
    for idx in range(len(mam)):
        padding_length = max_length - mam[idx].size(0)
        up_padding = (0, 0, 0, padding_length)  # 三角阵正下方补0
        mam[idx] = F.pad(mam[idx], up_padding, value=0)
        right_padding = (0, padding_length, 0, 0)  # 三角阵正右方补0
        mam[idx] = F.pad(mam[idx], right_padding, value=0)

    mam = torch.stack(mam).unsqueeze(1)  # stack: list[tensor] -> tensor
    return mam  # [batch_size, 1, max_len, max_len+M]


def generator_collate_fn(batch_data, tokenizer, real_batch_size, is_train):
    """
        function_inputs: {text1, text2}
        model_inputs: {nosiy_text1, text2}
        prompt: <bos>“<第一句>”的相似句是“<第二句>”<eos>
    """
    max_length, total_num = 400, 0
    prompts, lengths, attention_mask = [], [], []
    prompts_input_ids, lengths_input_ids, prompts_attention_mask = [], [], []
    for item in batch_data:
        if is_train:
            text2 = tokenizer(item['text2'], return_tensors='pt').input_ids
            noisy_text2 = noisy(x=text2, drop_prob=0.05, sub_prob=0.05, shuffle_dist=0, 
                                bos_token=50001, pad_token=50000, vocab_size=50176)
            item['text2'] = tokenizer.decode(noisy_text2.squeeze(), skip_special_tokens=True)

        # 反过来，让生成的文本作为text1, 原文本作为text2进行训练。
        prompt_text = '<bos>“' + item['text2'] + '”的相似句是“' + item['text1'] + '”'
        prompt = tokenizer(
            prompt_text.replace(' ', ''), return_tensors='pt').input_ids.squeeze()
        if len(prompt) > 400:
            prompt = torch.cat([prompt[:399], torch.tensor([43432, 50000])])  # 截断后拼上”<eos>
        
        # 由于sentence piece的原因，前面加“从而准确算出句子中text1_id的长度
        text1_ids = tokenizer(
            '“' + item['text1'] + '”', return_tensors='pt').input_ids.squeeze()[1:]
        length = torch.tensor([1] * (len(prompt) - len(text1_ids)) + \
            [len(text1_ids)] * len(text1_ids))
    
        if len(prompt) <= max_length:
            max_length -= len(prompt)

            mask = get_atten_mask(len(prompt))  # mask当前的下三角阵
            # 先在三角阵上方padding，因为目前不知道最大长度
            up_length = sum(len(p) for p in prompts)
            up_padding = (0, 0, up_length, 0)  # 三角阵正上方补0
            mask = F.pad(mask, up_padding, value=0)
            attention_mask.append(mask)

            prompts.append(prompt)
            total_num += 1
            lengths.append(length)

        #  还有可能没到最大长度数据就拼完了
        if len(prompt) > max_length or item == batch_data[-1]:
            prompts_input_ids.append(torch.cat(prompts))
            lengths_input_ids.append(torch.cat(lengths))

            # 知道最大长度后，在三角阵下方padding
            seq_length = sum(len(p) for p in prompts)
            for idx in range(len(attention_mask) - 1):  # 最后一个mask不需要padding
                down_length = seq_length - attention_mask[idx].size(0)
                down_padding = (0, 0, 0, down_length)  # 三角阵正下方补0
                attention_mask[idx] = F.pad(
                    attention_mask[idx], down_padding, value=0)
            prompts_attention_mask.append(torch.cat(attention_mask, dim=-1))
            if len(prompts_input_ids) >= real_batch_size:
                break

            # initialize
            max_length = 400
            prompts, lengths, attention_mask = [], [], []

    prompts_input_ids = pad_sequence([x for x in prompts_input_ids],
                                     batch_first=True, padding_value=50000)  # eos_token_id is 50000
    lengths_input_ids = pad_sequence([x for x in lengths_input_ids], 
                                    batch_first=True, padding_value=50000)

    max_seq_length = max(pam.size(0) for pam in prompts_attention_mask)
    prompts_attention_mask = padding_memory_mask(
        prompts_attention_mask, max_seq_length)

    return {
        'total_num': torch.tensor(total_num),
        'prompts_input_ids': prompts_input_ids,
        'lengths_input_ids': lengths_input_ids,
        'prompts_attention_mask': prompts_attention_mask,
    }


def discriminator_collate_fn(batch_data, tokenizer, is_train):
    """
        data: [CLS]<第一句>[SEP]<第二句>[SEP]
        label: 0/1
    """
    dis_text_input_ids, labels = [], []
    for item in batch_data:
        if is_train:
            text1 = tokenizer(item['text1'], return_tensors='pt').input_ids
            masked_text1 = mask_tokens(
                inputs=text1, tokenizer=tokenizer, mlm_prob=0.15).squeeze()
            text2 = tokenizer(item['text2'], return_tensors='pt').input_ids
            masked_text2 = mask_tokens(
                inputs=text2, tokenizer=tokenizer, mlm_prob=0.15).squeeze()
            input_ids = torch.cat((masked_text1, masked_text2[1:]), dim=0)
        else:
            dis_text = item['text1'] + '[SEP]' + item['text2']
            input_ids = tokenizer(dis_text, return_tensors='pt').input_ids.squeeze()
        
        if input_ids.size(0) > 203:
            continue

        dis_text_input_ids.append(input_ids)
        labels.append(torch.tensor(item['score'], dtype=torch.long))

    dis_text_input_ids = pad_sequence([x for x in dis_text_input_ids],
                                      batch_first=True, padding_value=0)

    return {
        'dis_text_input_ids': dis_text_input_ids,
        'labels': torch.stack(labels),
    }
