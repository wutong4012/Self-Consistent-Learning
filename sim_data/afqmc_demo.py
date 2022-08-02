import torch, os
import streamlit as st
import pandas as pd
from time import time
from transformers import BertForSequenceClassification, BertTokenizer


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

@st.cache
def load_model_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-330M-Similarity')
    model = BertForSequenceClassification.from_pretrained(
        'IDEA-CCNL/Erlangshen-Roberta-330M-Similarity', num_labels=2)
    pt_path = '/raid/wutong/discriminator.pt'

    new_dict = {}
    state_dict = torch.load(pt_path, map_location='cpu')['module']
    for k, v in state_dict.items():
        if any([i in k for i in ['module.discriminator.dis.']]):
            new_dict[k[len('module.discriminator.dis.'):]] = v
        else:
            continue
    model.load_state_dict(new_dict)
    model.cuda().eval()

    return model, tokenizer


model, tokenizer = load_model_tokenizer()
st.title('AFQMC Demo')
csv_file = st.file_uploader(label="请上传xlsx/xls文件, 表头为 {id sentence1  sentence2}。",
                            type=["xlsx","xls"])
st.write("例子:")
test_df = pd.read_excel('afqmc.xlsx')
st.write(test_df)

submit = st.button('提交')

if csv_file:
    df = pd.read_excel(csv_file.read())
    df['label'] = -1
    if submit:
        start = time()
        predictions = []
        bar = st.progress(0)
        for idx in range(len(df)):
            bar.progress(int(100 * (idx + 1) / len(df)))
            dis_text = df.iloc[idx]['sentence1'] + '[SEP]' + df.iloc[idx]['sentence2']
            input_ids = tokenizer(dis_text, return_tensors='pt').input_ids
            logits = model.forward(input_ids.cuda(), None).logits
            df.loc[idx, 'label'] = torch.argmax(logits, dim=1).item()
        used_time = time() - start

        st.write(f'使用时间为{used_time}s。')
        st.write('输出为excel文件, 表头为 {id sentence1 sentence2 label}')

        excel = df.to_csv(index=False).encode('utf_8_sig')
        st.download_button(label='下载excel文件',
                           data=excel,
                           file_name='afqmc_predict.csv')
