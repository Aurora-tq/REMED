import json
class LongTextDataloader(object):

    def __init__(self, filename: str, max_sub_sentence_len: int, batch_size: int, 
                    shuffle=False):
        """
        长文本dataloader，初始化函数。

        Args:
            filename (str): 数据集文件
            max_sub_sentence_len (int): 每个子句最大的长度限制
            batch_size (int): 一次返回多少句子
            shuffle (bool): 是否打乱数据集
        """
        self.texts = self.__read_file(filename)
        # assert len(self.texts) == len(self.labels), '[ERROR] texts count not equal label count.'
        self.start = 0
        self.end = len(self.texts)
        self.batch_size= batch_size
        self.max_sub_sentence_len = max_sub_sentence_len
        self.visit_order = [i for i in range(self.end)]
        if shuffle:
          random.shuffle(self.visit_order)
    def __read_file(self, filename: str) -> list:
        """
        将本地数据集读到数据加载器中。

        Args:
            filename (str): 数据集文件名

        Returns:
            [tuple] -> 文本列表，标签列表
        Returns:
            [list] -> 文本列表
        """
        texts = []
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                text = json.loads(line)
                # label, text = line.strip().split('\t')
                #texts.append(text['text']) 
                texts.append(text) 
                # labels.append(label)
        #print(type(texts))
        return texts

    def __split_long_text(self, text: str) -> list:
        """
        用于迭代器返回数据样本的时候将长文本切割为若干条。

        Args:
            text (str): 长文本, e.g. -> "我爱中国"
        
        Returns:
            [list] -> ["我爱", "中国"]（假设self.max_sub_sentence_len = 2）
        """
        sub_texts, start, length = [], 0, len(text)
        
        while start < length:
            sub_texts.append(text[start: start + self.max_sub_sentence_len])
            start += self.max_sub_sentence_len
        return sub_texts

    def __next__(self) -> dict:
        """
        迭代器，每次返回数据集中的一个样本，返回样本前会先将长文本切割为若干个短句子。

        Raises:
            StopIteration: [description]

        Returns:
            [dict] -> {
                'text': [sub_sentence 1, sub_sentence 2, ...],
                'label': 1
            }
        """
        if self.start < self.end:
            ret = self.start
            batch_end = ret + self.batch_size
            self.start += self.batch_size
            currents = self.visit_order[ret:batch_end]
            # print(currents)
            for item in self.texts:
                for c in currents:
                    #print(type(item['text']))
                    text = self.__split_long_text(item['text'])
            return {'text':text}  # 修改这行
        else:
            self.start = 0
            raise StopIteration
        # if self.start < self.end:
        #     ret = self.start
        #     batch_end = ret + self.batch_size
        #     self.start += self.batch_size
        #     currents = self.visit_order[ret: batch_end]
        #     return {'text': [self.__split_long_text(self.texts[c]) for c in currents]} 
        #     return {'text': [self.__split_long_text(self.texts[c]) for c in currents]}#, 'label': [int(self.labels[c]) for c in currents]
        # else:
        #     self.start = 0
        #     raise StopIteration
    
    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.labels)

    def forward(self, sub_texts: list, max_seq_len: int):
            """
            正向传播函数，将一段长文本中的所有N段子文本都过一遍backbone，得到N个pooled_output([CLS]过了一个tanh函数)，
            再将这N个pooled_output向量Pooling成一个768-dim的融合向量，融合向量中768-dim中的每一维都取这N个向量对应dim
            的最大值（MaxPooling），使用MaxPooling而非MeanPooling是因为BERT类的模型抽取的特征非常稀疏，Max-Pooling
            会保留突出的特征，Mean-Pooling会将特征拉平。

            Args:
                sub_texts (list[str]): batch个长文本被切成的所有子段列表 -> (batch, sub_text_num, sub_text_len)
                max_seq_len (int): tokenize之后的最大长度（文本长度+2）
            """
            sub_inputs = []
            for sub_text in sub_texts:                                                  # 一个batch的句子
                sub_idx = 0
                for sub in sub_text:                                                    # 一个句子中的子句
                    if sub_idx == self.max_sub_sentence_num:                            # 若达到最大子句数，则丢掉剩余的子句
                        break
                    encoded_inputs = self.tokenizer(text=sub, max_seq_len=max_seq_len)
                    input_ids = encoded_inputs["input_ids"]
                    token_type_ids = encoded_inputs["token_type_ids"]
                    sub_inputs.append([input_ids, token_type_ids])
                    sub_idx += 1
                while sub_idx < self.max_sub_sentence_num:                              # 若未达到最大子句数，则用空句子填满
                    sub_inputs.append([[], []])
                    sub_idx += 1

            sub_inputs = Tuple(                                                         # (batch*max_sub_setences, seq_len)
                Pad(axis=0, pad_val=self.tokenizer.pad_token_id),                       # input
                Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id)                   # segment
            )(sub_inputs)

            input_ids, token_type_ids = sub_inputs                                      # (batch*max_sub_setences, seq_len)
            input_ids, token_type_ids = paddle.to_tensor(input_ids), paddle.to_tensor(token_type_ids)
            
            sequence_output, pooled_output= self.backbone(input_ids, token_type_ids)    # sequence_output: (batch*max_sub_setences, seq_len, cls-dim)
                                                                                        # pooled_output: (batch*max_sub_setences, cls-dim)

            pooled_output = paddle.reshape(pooled_output, (-1, 1, self.max_sub_sentence_num, 768))  # (batch, 1, max_sub_setences, cls-dim)
            pooled = F.adaptive_max_pool2d(pooled_output, output_size=(1, 768)).squeeze()       # (batch, cls-dim)
            # pooled = F.adaptive_avg_pool2d(pooled_output, output_size=(1, 768)).squeeze()     # (batch, cls-dim)

            fc_out = self.fc(pooled)
            fc_out = self.activation(fc_out)
            output = self.output_layer(fc_out)                                          # (batch, 2)

            return output
if __name__ == "__main__":
    filename = '/mnt/workspace/pangtianqi/medical_kb_chatbot/data/selected_data.jsonl'
    max_sub_sentence_len = 512 # 每个子句最大长度限制
    batch_size = 1  # 一次返回多少句子
    shuffle = False  # 是否打乱数据集
    dataloader = LongTextDataloader(filename, max_sub_sentence_len, batch_size, shuffle)
    data_iter = iter(dataloader)
    sample = next(data_iter)
    text_list = []
    text_dict = {}
    for sample in dataloader:
        # text = sample['text'] 
        text_list.append(sample['text'])
    for index, text in enumerate(text_list):
        text_dict[index] ={"text": text}
    # print(type(text_dict))
    with open("Dataset2_processed.json", "w") as f:
        json.dump(text_dict, f)
    