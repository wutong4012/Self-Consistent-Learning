import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from data_utlis.noisy_input_ids import mask_tokens, noisy


def padding_dis_mask(mam: list, max_length):
    for idx in range(len(mam)):
        padding_length = max_length - mam[idx].size(0)
        up_padding = (0, 0, 0, padding_length)  # 三角阵正下方补0
        mam[idx] = F.pad(mam[idx], up_padding, value=0)
        right_padding = (0, padding_length, 0, 0)  # 三角阵正右方补0
        mam[idx] = F.pad(mam[idx], right_padding, value=0)

    mam = torch.stack(mam)  # stack: list[tensor] -> tensor
    return mam  # [batch_size, max_len, max_len+M]


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
        prompt: “<第一句>”的相似句是“<第二句>”<eos>
    """
    max_length, total_num = 400, 0
    prompts, lengths, attention_mask = [], [], []
    prompts_input_ids, lengths_input_ids, prompts_attention_mask = [], [], []
    for item in batch_data:
        if is_train:
            text1 = tokenizer(item['sentence1'], return_tensors='pt').input_ids
            noisy_text1 = noisy(x=text1, drop_prob=0.05, sub_prob=0.05, shuffle_dist=0, 
                                bos_token=50001, pad_token=50000, vocab_size=50176)
            item['sentence1'] = tokenizer.decode(noisy_text1.squeeze(), skip_special_tokens=True)

        prompt_text = '“' + item['sentence1'] + '”的相似句是“' + item['sentence2'] + '”'
        prompt = tokenizer(
            prompt_text.replace(' ', ''), return_tensors='pt').input_ids.squeeze()
        if len(prompt) > 400:
            prompt = torch.cat([prompt[:399], torch.tensor([43432, 50000])])  # 截断后拼上”<eos>
        
        # 由于sentence piece的原因，前面加“从而准确算出句子中text2_id的长度
        text2_ids = tokenizer(
            '“' + item['sentence2'] + '”', return_tensors='pt').input_ids.squeeze()[1:]
        length = torch.tensor([1] * (len(prompt) - len(text2_ids)) + \
            [len(text2_ids)] * len(text2_ids))
    
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
                                     batch_first=True, padding_value=1)

    max_seq_length = max(pam.size(0) for pam in prompts_attention_mask)
    prompts_attention_mask = padding_memory_mask(
        prompts_attention_mask, max_seq_length)

    print('***************************************')
    print('total_num=', total_num)
    print('***************************************')

    return {
        'total_num': torch.tensor(total_num),
        'prompts_input_ids': prompts_input_ids,
        'lengths_input_ids': lengths_input_ids,
        'prompts_attention_mask': prompts_attention_mask,
    }


def discriminator_collate_fn(batch_data, tokenizer):
    max_length = 0
    input_ids, token_type_ids, attention_mask, position_ids, mlmlabels = [], [], [], [], []
    clslabels, clslabels_mask, mlmlabels_mask, sentence1, sentence2, label_idx = [], [], [], [], [], []
    for item in batch_data:
        max_length = max(max_length, item['attention_mask'].size(0))
        input_ids.append(item['input_ids'])
        token_type_ids.append(item['token_type_ids'])
        attention_mask.append(item['attention_mask'])
        position_ids.append(item['position_ids'])
        mlmlabels.append(item['mlmlabels'])
        clslabels.append(item['clslabels'])
        clslabels_mask.append(item['clslabels_mask'])
        mlmlabels_mask.append(item['mlmlabels_mask'])
        # label_idx.append(item['label_idx'])
        sentence1.append(item['sentence1'])
        sentence2.append(item['sentence2'])
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attention_mask = padding_dis_mask(attention_mask, max_length)
    position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
    mlmlabels = pad_sequence(mlmlabels, batch_first=True, padding_value=0)
    clslabels_mask = pad_sequence(clslabels_mask, batch_first=True, padding_value=-10000)
    mlmlabels_mask = pad_sequence(mlmlabels_mask, batch_first=True, padding_value=0)
        
    return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "mlmlabels": mlmlabels,
            "clslabels": torch.stack(clslabels),
            "clslabels_mask": clslabels_mask,
            "mlmlabels_mask": mlmlabels_mask,
            # 'label_idx': torch.stack(label_idx),
            'sentence1': sentence1,
            'sentence2': sentence2
        }


def generator_en_collate_fn(batch_data, tokenizer, is_train):
    """
        function_inputs: {text1, text2}
        model_inputs: {nosiy_text1, text2}
        prompt: "<text1>" is similar to "<text2>"
    """
    input_ids, lengths = [], []
    for item in batch_data:
        if is_train:
            text1 = tokenizer(item['text1'], return_tensors='pt').input_ids
            noisy_text1 = noisy(x=text1, drop_prob=0, sub_prob=0.05, shuffle_dist=0, 
                                bos_token=2, pad_token=1, vocab_size=50272)
            try:
                item['text1'] = tokenizer.decode(noisy_text1.squeeze(), skip_special_tokens=True)
            except TypeError:
                pass
        if item['text2'] == 'general':
            prompt_text = item['text1']
            prompt = tokenizer(prompt_text, return_tensors='pt')
            # 太长只取前400个token
            prompt_input_ids = torch.cat((prompt.input_ids.squeeze()[1:400], torch.tensor([2])))
            length = torch.tensor([1 / len(prompt_input_ids)] * len(prompt_input_ids))
            
        else:
            prompt_text = '"' + item['text1'] + '" is similar to "' + item['text2'] + '"'
            prompt = tokenizer(prompt_text, return_tensors='pt')
            if len(prompt.input_ids.squeeze()) > 512:
                continue
            # 因为<bos>和<eos>的token id一样，所以去掉自动添加的<bos>, 并手动添加<eos>
            prompt_input_ids = torch.cat((prompt.input_ids.squeeze()[1:], torch.tensor([2])))

            # 自动加了<bos>, 等效于手动加了<eos>, 长度不变
            text2_ids = tokenizer(item['text2'] + '"', return_tensors='pt').input_ids.squeeze()
            length = torch.tensor(
                [0] * (len(prompt_input_ids) - len(text2_ids)) + [1 / len(text2_ids)] * len(text2_ids))

        input_ids.append(prompt_input_ids)
        lengths.append(length)
        
    input_ids = pad_sequence([x for x in input_ids], batch_first=True, padding_value=1)
    lengths = pad_sequence([x for x in lengths], batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'lengths': lengths
    }