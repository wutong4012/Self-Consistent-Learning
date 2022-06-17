import datasets
import csv, tqdm, jsonlines, glob, os
import random
from concurrent.futures import ProcessPoolExecutor


_JSON_DATA_PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/sim_json_data'
_CACHE_DATA_PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/sim_cache_data'
_HUGGINGFACE_CACHE = '/cognitive_comp/wutong/source/data_base/huggingface-cache/'
_WUDAO_DATA_PATH = '/cognitive_comp/wutong/source/data_base/wudao_sentences/'


def load_wudao_data():
    cache_dict_paths = glob.glob(os.path.join(_WUDAO_DATA_PATH, '*'))
    wudao_ds, res, wudao_sentences = [], [], []
    p = ProcessPoolExecutor(max_workers=1)
    
    for path in cache_dict_paths:
        res.append(p.submit(datasets.load_from_disk, path))
    p.shutdown(wait=True)
    for future in res:
        wudao_ds.append(future.result())
    wudao_ds = datasets.concatenate_datasets(wudao_ds)
    
    for idx in tqdm.tqdm(range(wudao_ds.num_rows), desc="Concat_Data"):
        if wudao_ds[idx]['sentence_list'] is not None and wudao_ds[idx]['sentence_list'] != []:
            wudao_sentences.extend(wudao_ds[idx]['sentence_list'])
    
    print(f'There ara total {len(wudao_sentences)} sentences !')
    random_list = random.sample(range(len(wudao_sentences)), 10)
    for i in random_list:
        print("Examples: {}".format(wudao_sentences[i]))
    
    return wudao_sentences


class SimPairReader():
    def __init__(self, wudao_sentences):
        path = self.PATH
        data_list = []
        csv.field_size_limit(10000000)
        print(f"Start reading {path}")
        with open(path) as file:
            reader = csv.reader(file)
            progress_bar = tqdm.tqdm()
            while True:
                progress_bar.update()
                neg_sent1 = wudao_sentences[self.ids0]
                neg_sent2 = wudao_sentences[self.ids1]
                self.ids0 += 2
                self.ids1 += 2
                try:
                    line = next(reader)
                    if '\n' in line[0]:  # 单行含有N个样本
                        lines = ''.join(line).split('\n')
                        for line in lines:
                            dict_data = self.process_line(line, neg_sent1, neg_sent2)  # list
                            if dict_data is not None:
                                data_list.extend(dict_data)
                        continue
                except StopIteration:
                    break
                dict_data = self.process_line(line, neg_sent1, neg_sent2)
                if dict_data is not None:
                    data_list.extend(dict_data)
            print('end of reading {}'.format(path))
            progress_bar.close()
        
        print(f'Total {len(data_list)} samples are save to 0{self.json_name_id}.json !')
        with jsonlines.open(_JSON_DATA_PATH + f'/0{self.json_name_id}.json', 'w') as json_file:
            for dict_item in data_list:
                json_file.write(dict_item)
            json_file.close()
        
    @classmethod
    def is_sim(cls, label):
        return int(label) == 1

    @classmethod
    def split(cls, data):
        return ''.join(data).split('\t')

    @classmethod
    def process_line(cls, data, neg_sent1, neg_sent2):
        # data, list, only one element ['a \t b \t label']
        split_str = cls.split(data)
        text_a = split_str[0]
        text_b = split_str[1]
        
        """
        dirty data:
        ['可以申请延期还款日吗？\t如果还款日那天不能及时还款，能否延后几日再还？有没有影响？']
        """
        if len(split_str) < 3:
            return None
        else:
            is_sim = cls.is_sim(split_str[2])  # bool
        
        if is_sim:
            dict_data = (
                {'text1': text_a, 'text2': text_b, 'score': 1},
                {'text1': text_b, 'text2': text_a, 'score': 1},
                {'text1': text_a, 'text2': neg_sent1, 'score': 0},
                {'text1': text_b, 'text2': neg_sent2, 'score': 0},
            )
            return dict_data
        
        return None


class atec(SimPairReader):
    json_name_id = '0'
    ids0, ids1 = 556706*2, 556706*2+1
    PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/similar_raw_data/ATEC/ATEC.data'

class atec_ccks(SimPairReader):
    json_name_id = '1'
    ids0, ids1 = 556706*3, 556706*3+1
    PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/similar_raw_data/ATEC_CCKS/ATEC_CCKS.data'

class bq(SimPairReader):
    json_name_id = '2'
    ids0, ids1 = 556706*4, 556706*4+1
    PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/similar_raw_data/BQ/BQ.data'

class ccks2018(SimPairReader):
    json_name_id = '3'
    ids0, ids1 = 556706*5, 556706*5+1
    PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/similar_raw_data/CCKS_2018_3/CCKS_2018.data'

class lcqmc(SimPairReader):
    json_name_id = '4'
    ids0, ids1 = 556706*6, 556706*6+1
    PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/similar_raw_data/LCQMC/LCQMC.data'

class pawsx(SimPairReader):
    json_name_id = '5'
    ids0, ids1 = 556706*7, 556706*7+1
    PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/similar_raw_data/PAWSX/PAWSX.data'

class sts_b(SimPairReader):
    json_name_id = '6'
    ids0, ids1 = 556706*8, 556706*8+1
    PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/similar_raw_data/STS_B/STS_B.data'

    @classmethod
    def is_sim(cls, label):
        return int(label) >= 4

class idea_anno(SimPairReader):
    json_name_id = '7'
    ids0, ids1 = 0, 1
    PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/similar_raw_data/idea_anno/similar_stence_sep_4#.txt'

    @classmethod
    def split(cls, data):
        return ''.join(data).split('|##|')

class idea_anno2(SimPairReader):
    json_name_id = '8'
    ids0, ids1 = 556706*1, 556706*1+1
    PATH = '/cognitive_comp/wutong/source/data_base/similarity_data/similar_raw_data/idea_anno/20211125_20211201/20211125_20211201.txt'

    @classmethod
    def split(cls, data):
        return ''.join(data).split('|##|')
    

feats = datasets.Features({"text1": datasets.Value('string'), 
                           "text2": datasets.Value('string'),
                           "score": datasets.Value('int8')})
def _generate_cache_arrow(index, path):
    print('saving dataset shard {}'.format(index))
    ds = (datasets.load_dataset('json', data_files=path,
                                cache_dir=_HUGGINGFACE_CACHE,
                                features=feats)['train'])
    ds.save_to_disk(os.path.join(_CACHE_DATA_PATH, f'0{index}'))
    return 'saving dataset shard {} done'.format(index)


def generate_cache_arrow(num_proc=1) -> None:
    '''
    生成HF支持的缓存文件，加速后续的加载
    '''
    data_dict_paths = glob.glob(_JSON_DATA_PATH + '/*.json')
    print(data_dict_paths)
    
    p = ProcessPoolExecutor(max_workers=num_proc)
    res = []

    for index, path in enumerate(data_dict_paths):
        res.append(p.submit(_generate_cache_arrow, index, path))

    p.shutdown(wait=True)
    for future in res:
        print(future.result(), flush=True)


if __name__ == '__main__':
    if not os.path.exists(_JSON_DATA_PATH):
        os.makedirs(_JSON_DATA_PATH)
    if not os.path.exists(_CACHE_DATA_PATH):
        os.makedirs(_CACHE_DATA_PATH)
    if not os.path.exists(_HUGGINGFACE_CACHE):
        os.makedirs(_HUGGINGFACE_CACHE)
    
    # wudao_sentences = load_wudao_data()
    # named_corpora = {
    #     # 'idea_anno': idea_anno(wudao_sentences),
    #     # 'idea_anno2': idea_anno2(wudao_sentences),
    #     'atec': atec(wudao_sentences),
    #     'atec_ccks': atec_ccks(wudao_sentences),
    #     "bq": bq(wudao_sentences),
    #     "ccks2018": ccks2018(wudao_sentences),
    #     "lcqmc": lcqmc(wudao_sentences),
    #     "pawsx": pawsx(wudao_sentences),
    #     "sts_b": sts_b(wudao_sentences),
    # }

    print('Starting Save to Cache...')
    generate_cache_arrow()
