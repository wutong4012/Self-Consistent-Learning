import torch
import datasets

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from torch.nn.utils.rnn import pad_sequence

from data_utlis.sim_data_collate import padding_dis_mask
from data_utlis.sim_gen_dataset import SimGanDataset, preprocess
from model_utils.sim_gen_model import Discriminator

def dis_pred_collate(batch_data, tokenizer):
    max_length = 0
    input_ids, token_type_ids, attention_mask, position_ids = [], [], [], []
    clslabels_mask, sentence1, sentence2, labels, label_idx = [], [], [], [], []
    for item in batch_data:
        max_length = max(max_length, item['attention_mask'].size(0))
        input_ids.append(item['input_ids'])
        token_type_ids.append(item['token_type_ids'])
        attention_mask.append(item['attention_mask'])
        position_ids.append(item['position_ids'])
        clslabels_mask.append(item['clslabels_mask'])
        sentence1.append(item['sentence1'])
        sentence2.append(item['sentence2'])
        labels.append(item['label'])
        label_idx.append(item['label_idx'])
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attention_mask = padding_dis_mask(attention_mask, max_length)
    position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
    clslabels_mask = pad_sequence(clslabels_mask, batch_first=True, padding_value=-10000)
        
    return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "clslabels_mask": clslabels_mask,
            'label_idx': torch.stack(label_idx),
            'sentence1': sentence1,
            'sentence2': sentence2,
            'labels': labels
        }


def check_acc(config, last_acc=0):
    dis_tokenizer = AutoTokenizer.from_pretrained(config.dis_model_path)

    test_data = datasets.Dataset.from_json(config.data_path + '/test_public.json')
    test_data = test_data.map(preprocess)
    test_dataset = SimGanDataset(data=test_data, tokenizer=dis_tokenizer, test=True)

    def collate_fn(batch_data):
        return dis_pred_collate(batch_data, dis_tokenizer)
    dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    all_labels, all_preds = [], []
    discriminator = Discriminator(config, dis_tokenizer)
    discriminator.cuda().eval()
    with torch.no_grad():
        for batch in dataloader:
            torch.cuda.empty_cache()
            prob = discriminator.forward(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                token_type_ids=batch['token_type_ids'].cuda(),
                position_ids=batch['position_ids'].cuda(),
                clslabels_mask=batch['clslabels_mask'].cuda(),
                bt_label_idx=batch['label_idx'].cuda()
            )
            
            predictions = torch.argmax(prob, dim=-1).tolist()
            all_labels.extend(batch['labels'])
            all_preds.extend(predictions)
            
        acc_result = accuracy_score(all_labels, all_preds)
    
    if acc_result > last_acc:
        return True, acc_result
    else:
        return False, last_acc
