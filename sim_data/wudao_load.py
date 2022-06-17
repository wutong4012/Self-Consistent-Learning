import re, os
import string

from concurrent.futures import ProcessPoolExecutor


_MAX_SENTENCE_LENGTH = 100
_TRAIN_SHARD_PART = 100
_NUM_PROC = 1

# 缓存文件
_CACHE_DATA_PATH = '/cognitive_comp/wutong/source/data_base/wudao_sentences/'


remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def filter_str(sentence):
    sentence = re.sub(remove_nota, '', sentence)
    sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()

def judge_language(s):
    s = filter_str(s)
    result = []
    s = re.sub('[0-9]', '', s).strip()
    # unicode english
    re_words = re.compile(u"[a-zA-Z]")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub('[a-zA-Z]', '', s).strip()
    if len(res) > 0:
        result.append('en')
    if len(res2) <= 0:
        return 'en'

    # unicode chinese
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\u4e00-\u9fa5]+", '', s).strip()
    if len(res) > 0:
        result.append('zh')
    if len(res2) <= 0:
        return 'zh'

    # unicode korean
    re_words = re.compile(u"[\uac00-\ud7ff]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\uac00-\ud7ff]+", '', s).strip()
    if len(res) > 0:
        result.append('ko')
    if len(res2) <= 0:
        return 'ko'

    # unicode japanese katakana and unicode japanese hiragana
    re_words = re.compile(u"[\u30a0-\u30ff\u3040-\u309f]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\u30a0-\u30ff\u3040-\u309f]+", '', s).strip()
    if len(res) > 0:
        result.append('ja')
    if len(res2) <= 0:
        return 'ja'
    return ','.join(result)


def _generate_cache_arrow(index, ds):
    print('saving dataset shard {}'.format(index))
    ds.save_to_disk(os.path.join(_CACHE_DATA_PATH, 'part_{}'.format(index)))
    return 'saving dataset shard {} done'.format(index)


def generate_arrow_cache(num_proc) -> None:
    '''
    读取wudao_180g原始数据，并进行切句，切句后生成datasets
    同时利用seed 42做shuffle 缓存下来
    '''
    import sys
    sys.path.append('../../')
    from fs_datasets import load_dataset
    ds = load_dataset('wudao_180g', num_proc=100)
    ds = ds['train'].train_test_split(train_size=0.99, test_size=0.01, seed=42)
    ds = ds['test'].train_test_split(train_size=0.99, test_size=0.01, seed=42)
    print(ds)

    def _clean_and_split(example):
        langs_type = judge_language(example['text'])
        
        # 删去非中文text
        if langs_type == 'zh':
            # split_sentence = re.split(r',|;|!|\?|\.|，|。|；|！|？|…', example['text']) # 英文?和.需要转义符\
            split_sentence = re.split(r'!|\?|\.|。|！|？|…', example['text'])  # 去掉逗号分号
            for idx in range(len(split_sentence)-1, -1, -1): # reversed order.
                # delete too short and too long.
                if len(split_sentence[idx]) >= _MAX_SENTENCE_LENGTH or len(split_sentence[idx]) <= 10:
                    del split_sentence[idx]
            
            if split_sentence is not None:
                return {'sentence_list': split_sentence}
        
        return {'sentence_list': None}

    ds_sentences = ds.map(
        _clean_and_split,
        num_proc=num_proc,
        remove_columns=['text'])

    p = ProcessPoolExecutor(max_workers=num_proc)
    res = []
    train_shard_part = _TRAIN_SHARD_PART
    for i in range(0, train_shard_part):
        res.append(p.submit(_generate_cache_arrow, i,
                            ds_sentences['train'].shard(train_shard_part, i)))

    p.shutdown(wait=True)
    for future in res:
        print(future.result(), flush=True)

    print('done')


if __name__ == '__main__':
    if not os.path.exists(_CACHE_DATA_PATH):
        os.makedirs(_CACHE_DATA_PATH)

    generate_arrow_cache(num_proc=_NUM_PROC)
