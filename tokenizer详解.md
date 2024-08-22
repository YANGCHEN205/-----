# 大模型基础组件 - Tokenizer

Tokenizer分词算法是NLP大模型最基础的组件，基于Tokenizer可以将文本转换成独立的token列表，进而转换成输入的向量成为计算机可以理解的输入形式。本文将对分词器进行系统梳理，包括分词模型的演化路径，可用的工具，并**手推**每个tokenizer的具体实现。

**速览**

1. 根据不同的切分粒度可以把tokenizer分为: 基于词的切分，基于字的切分和基于subword的切分。 基于subword的切分是目前的主流切分方式。
2. subword的切分包括: BPE(/BBPE), WordPiece 和 Unigram三种分词模型。其中WordPiece可以认为是一种特殊的BPE。
3. 完整的分词流程包括：文本归一化，预切分，基于分词模型的切分，后处理。
4. SentencePiece是一个分词工具，内置BEP等多种分词方法，基于Unicode编码并且将空格视为特殊的token。这是当前大模型的主流分词方案。

| 分词方法  | 典型模型                                                               |
| --------- | ---------------------------------------------------------------------- |
| BPE       | GPT, GPT-2, GPT-J, GPT-Neo, RoBERTa, BART, LLaMA, ChatGLM-6B, Baichuan |
| WordPiece | BERT, DistilBERT，MobileBERT                                           |
| Unigram   | AlBERT, T5, mBART, XLNet                                               |

## **1. 基于subword的切分**

基于subword的切分能很好平衡基于词切分和基于字切分的优缺点，也是目前主流最主流的切分方式。基于词和字的切分都会存在一定的问题，直接应用的效果比较差。

基于词的切分，会造成:

* 词表规模过大
* 一定会存在UNK，造成信息丢失
* 不能学习到词缀之间的关系，例如：dog与dogs，happy与unhappy

基于字的切分，会造成:

* 每个token的信息密度低
* 序列过长，解码效率很低

所以基于词和基于字的切分方式是两个极端，其优缺点也是互补的。而折中的subword就是一种相对平衡的方案。

subword的基本切分原则是：

* 高频词依旧切分成完整的整词
* 低频词被切分成有意义的子词，例如 dogs => [dog, ##s]

基于subword的切分可以实现：

* 词表规模适中，解码效率较高
* 不存在UNK，信息不丢失
* 能学习到词缀之间的关系

基于subword的切分包括：BPE，WordPiece 和 Unigram 三种分词模型。

## **2.切分流程**

Tokenizer包括训练和推理两个环节。训练阶段指得是从语料中获取一个分词器模型。推理阶段指的是给定一个句子，基于分词模型切分成一连串的token。基本的流程如图所示，包括归一化，预分词，基于分词模型的切分，后处理4个步骤。

![1724321316330](image/tokenizer详解/1724321316330.png)

### **2.1. 归一化**

这是最基础的文本清洗，包括删除多余的换行和空格，转小写，移除音调等。例如：

```text
input: Héllò hôw are ü?
normalization: hello how are u?
```

HuggingFace tokenizer的实现： [https://**huggingface.co/docs/tok**](https://link.zhihu.com/?target=https%3A//huggingface.co/docs/tokenizers/api/normalizers)

### **2.2. 预分词**

预分词阶段会把句子切分成更小的“词”单元。可以基于空格或者标点进行切分。 不同的tokenizer的实现细节不一样。例如:

```plaintext
input: Hello, how are  you?

pre-tokenize:
[BERT]: [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]

[GPT2]: [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)), ('?', (19, 20))]

[t5]: [('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))] 
```

可以看到BERT的tokenizer就是直接基于空格和标点进行切分。 GPT2也是基于空格和标签，但是空格会保留成特殊字符“Ġ”。 T5则只基于空格进行切分，标点不会切分。并且空格会保留成特殊字符"▁"，并且句子开头也会添加特殊字符"▁"。

预分词的实现： [https://**huggingface.co/docs/tok**](https://link.zhihu.com/?target=https%3A//huggingface.co/docs/tokenizers/api/pre-tokenizers)

### **2.3. 基于分词模型的切分**

这里指的就是不同分词模型具体的切分方式。分词模型包括：BPE，WordPiece 和 Unigram 三种分词模型。

分词模型的实现： [https://**huggingface.co/docs/tok**](https://link.zhihu.com/?target=https%3A//huggingface.co/docs/tokenizers/api/models)

### **2.4. 后处理**

后处理阶段会包括一些特殊的分词逻辑，例如添加sepcial token：[CLS],[SEP]等。

后处理的实现： [https://**huggingface.co/docs/tok**](https://link.zhihu.com/?target=https%3A//huggingface.co/docs/tokenizers/api/post-processors)

## **3.BPE**

Byte-Pair Encoding(BPE)是最广泛采用的subword分词器。

* [训练方法](https://zhida.zhihu.com/search?q=%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95)：从字符级的小词表出发，训练产生合并规则以及一个词表
* [编码方法](https://zhida.zhihu.com/search?q=%E7%BC%96%E7%A0%81%E6%96%B9%E6%B3%95)：将文本切分成字符，再应用训练阶段获得的合并规则
* 经典模型：GPT, GPT-2, RoBERTa, BART, LLaMA, ChatGLM等

### **3.1. 训练阶段**

在训练环节，目标是给定语料，通过训练算法，生成合并规则和词表。 BPE算法是从一个字符级别的词表为基础，合并pair并添加到词表中，逐步形成大词表。合并规则为选择相邻pair词频最大的进行合并。

下面我们进行手工的实现。

假定训练的语料(已归一化处理)为4个句子。

```plaintext
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

首先进行预切分处理。这里采用gpt2的预切分逻辑。 具体会按照空格和标点进行切分，并且空格会保留成特殊的字符“Ġ”。

```python
from transformers import AutoTokenizer

# init pre tokenize function
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
pre_tokenize_function = gpt2_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str

# pre tokenize
pre_tokenized_corpus = [pre_tokenize_str(text) for text in corpus]
```

获得的pre_tokenized_corpus如下，每个单元分别为[word, (start_index, end_index)]

```plaintext
[
    [('This', (0, 4)), ('Ġis', (4, 7)), ('Ġthe', (7, 11)), ('ĠHugging', (11, 19)), ('ĠFace', (19, 24)), ('ĠCourse', (24, 31)), ('.', (31, 32))], 
    [('This', (0, 4)), ('Ġchapter', (4, 12)), ('Ġis', (12, 15)), ('Ġabout', (15, 21)), ('Ġtokenization', (21, 34)), ('.', (34, 35))], 
    [('This', (0, 4)), ('Ġsection', (4, 12)), ('Ġshows', (12, 18)), ('Ġseveral', (18, 26)), ('Ġtokenizer', (26, 36)), ('Ġalgorithms', (36, 47)), ('.', (47, 48))], 
    [('Hopefully', (0, 9)), (',', (9, 10)), ('Ġyou', (10, 14)), ('Ġwill', (14, 19)), ('Ġbe', (19, 22)), ('Ġable', (22, 27)), ('Ġto', (27, 30)), ('Ġunderstand', (30, 41)), ('Ġhow', (41, 45)), ('Ġthey', (45, 50)), ('Ġare', (50, 54)), ('Ġtrained', (54, 62)), ('Ġand', (62, 66)), ('Ġgenerate', (66, 75)), ('Ġtokens', (75, 82)), ('.', (82, 83))]
]
```

进一步统计每个整词的词频

```python
word2count = defaultdict(int)
for split_text in pre_tokenized_corpus:
    for word, _ in split_text:
        word2count[word] += 1
```

获得word2count如下

```plaintext
`defaultdict(<class 'int'>, {'This': 3, 'Ġis': 2, 'Ġthe': 1, 'ĠHugging': 1, 'ĠFace': 1, 'ĠCourse': 1, '.': 4, 'Ġchapter': 1, 'Ġabout': 1, 'Ġtokenization': 1, 'Ġsection': 1, 'Ġshows': 1, 'Ġseveral': 1, 'Ġtokenizer': 1, 'Ġalgorithms': 1, 'Hopefully': 1, ',': 1, 'Ġyou': 1, 'Ġwill': 1, 'Ġbe': 1, 'Ġable': 1, 'Ġto': 1, 'Ġunderstand': 1, 'Ġhow': 1, 'Ġthey': 1, 'Ġare': 1, 'Ġtrained': 1, 'Ġand': 1, 'Ġgenerate': 1, 'Ġtokens': 1})`
```

因为BPE是从字符级别的小词表，逐步合并成大词表，所以需要先获得字符级别的小词表。

```python
vocab_set = set()
for word in word2count:
    vocab_set.update(list(word))
vocabs = list(vocab_set)
```


获得的初始小词表vocabs如下:

```plaintext
['i', 't', 'p', 'o', 'r', 'm', 'e', ',', 'y', 'v', 'Ġ', 'F', 'a', 'C', 'H', '.', 'f', 'l', 'u', 'c', 'T', 'k', 'h', 'z', 'd', 'g', 'w', 'n', 's', 'b']
```



基于小词表就可以对每个整词进行切分

```plaintext
word2splits = {word: [c for c in word] for word in word2count}

'This': ['T', 'h', 'i', 's'], 
'Ġis': ['Ġ', 'i', 's'], 
'Ġthe': ['Ġ', 't', 'h', 'e'], 
...
'Ġand': ['Ġ', 'a', 'n', 'd'], 
'Ġgenerate': ['Ġ', 'g', 'e', 'n', 'e', 'r', 'a', 't', 'e'], 
'Ġtokens': ['Ġ', 't', 'o', 'k', 'e', 'n', 's']
```


基于word2splits统计vocabs中相邻两个pair的词频pair2count

```python
def _compute_pair2score(word2splits, word2count):
    pair2count = defaultdict(int)
    for word, word_count in word2count.items():
        split = word2splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair2count[pair] += word_count
    return pair2count
```

获得pair2count如下：

```plaintext
defaultdict(<class 'int'>, {('T', 'h'): 3, ('h', 'i'): 3, ('i', 's'): 5, ('Ġ', 'i'): 2, ('Ġ', 't'): 7, ('t', 'h'): 3, ..., ('n', 's'): 1})
```

统计当前频率最高的相邻pair

```python
def _compute_most_score_pair(pair2count):
    best_pair = None
    max_freq = None
    for pair, freq in pair2count.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    return best_pair
```

经过统计，当前频率最高的pair为: ('Ġ', 't')， 频率为7次。 将('Ġ', 't')合并成一个词并添加到词表中。同时在合并规则中添加('Ġ', 't')这条合并规则。

```python
merge_rules = []
best_pair = self._compute_most_score_pair(pair2score)
vocabs.append(best_pair[0] + best_pair[1])
merge_rules.append(best_pair)
```

此时的vocab词表更新成:

```plaintext
['i', 't', 'p', 'o', 'r', 'm', 'e', ',', 'y', 'v', 'Ġ', 'F', 'a', 'C', 'H', '.', 'f', 'l', 'u', 'c', 'T', 'k', 'h', 'z', 'd', 'g', 'w', 'n', 's', 'b', 
'Ġt']
```

根据更新后的vocab重新对word2count进行切分。具体实现上，可以直接在旧的word2split上应用新的合并规则('Ġ', 't')

```python
def _merge_pair(a, b, word2splits):
    new_word2splits = dict()
    for word, split in word2splits.items():
        if len(split) == 1:
            new_word2splits[word] = split
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        new_word2splits[word] = split
    return new_word2splits
```

从而获得新的word2split

```plaintext
{'This': ['T', 'h', 'i', 's'], 
'Ġis': ['Ġ', 'i', 's'], 
'Ġthe': ['Ġt', 'h', 'e'], 
'ĠHugging': ['Ġ', 'H', 'u', 'g', 'g', 'i', 'n', 'g'],
...
'Ġtokens': ['Ġt', 'o', 'k', 'e', 'n', 's']}
```

可以看到新的word2split中已经包含了新的词"Ġt"。

重复上述循环直到整个词表的大小达到预先设定的词表大小。

```python
while len(vocabs) < vocab_size:
    pair2score = self._compute_pair2score(word2splits, word2count)
    best_pair = self._compute_most_score_pair(pair2score)
    vocabs.append(best_pair[0] + best_pair[1])
    merge_rules.append(best_pair)
    word2splits = self._merge_pair(best_pair[0], best_pair[1], word2splits)
```

假定最终词表的大小为50，经过上述迭代后我们获得的词表和合并规则如下：

```plaintext
vocabs = ['i', 't', 'p', 'o', 'r', 'm', 'e', ',', 'y', 'v', 'Ġ', 'F', 'a', 'C', 'H', '.', 'f', 'l', 'u', 'c', 'T', 'k', 'h', 'z', 'd', 'g', 'w', 'n', 's', 'b', 'Ġt', 'is', 'er', 'Ġa', 'Ġto', 'en', 'Th', 'This', 'ou', 'se', 'Ġtok', 'Ġtoken', 'nd', 'Ġis', 'Ġth', 'Ġthe', 'in', 'Ġab', 'Ġtokeni', 'Ġtokeniz']

merge_rules = [('Ġ', 't'), ('i', 's'), ('e', 'r'), ('Ġ', 'a'), ('Ġt', 'o'), ('e', 'n'), ('T', 'h'), ('Th', 'is'), ('o', 'u'), ('s', 'e'), ('Ġto', 'k'), ('Ġtok', 'en'), ('n', 'd'), ('Ġ', 'is'), ('Ġt', 'h'), ('Ġth', 'e'), ('i', 'n'), ('Ġa', 'b'), ('Ġtoken', 'i'), ('Ġtokeni', 'z')]
```

至此我们就根据给定的语料完成了BPE分词器的训练。


**3.2. 推理阶段**

在推理阶段，给定一个句子，我们需要将其切分成一个token的序列。 具体实现上需要先对句子进行预分词并切分成字符级别的序列，然后根据合并规则进行合并。


```python
def tokenize(self, text: str) -> List[str]:
    # pre tokenize
    words = [word for word, _ in self.pre_tokenize_str(text)]
    # split into char level
    splits = [[c for c in word] for word in words]
    # apply merge rules
    for merge_rule in self.merge_rules:
        for index, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == merge_rule[0] and split[i + 1] == merge_rule[1]:
                    split = split[:i] + ["".join(merge_rule)] + split[i + 2:]
                else:
                    i += 1
            splits[index] = split
    return sum(splits, [])
```


### **3.3. BBPE**

2019年提出的Byte-level BPE (BBPE)算法是上面BPE算法的进一步升级。具体参见：[Neural Machine Translation with Byte-Level Subwords](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.03341.pdf)。 核心思想是用byte来构建最基础的词表而不是字符。首先将文本按照UTF-8进行编码，每个字符在UTF-8的表示中占据1-4个byte。 在byte序列上再使用BPE算法，进行byte level的相邻合并。编码形式如下图所示：

![1724324281694](image/tokenizer详解/1724324281694.png)


通过这种方式可以更好的处理踣语言和不常见字符的特殊问题(例如，颜文字+)，相比传统的BPE更节省词表空间（同等词表大小效果更好），每个tokentb能获得更充分的训绕。

但是在解码阶段，一个byte序列可能解码后不是一个合法的字符序列，这里需要采用动态规划的算法进行解码，便其能解码出屈可能各的合法字符。具体算法如下：假定 $f(k)$ 表示字符序列 $B_{1, k}$ 最大能解码的合法字符数量， $f(k)$ 有最优的子结构:

$$
f(k)=\max _{t=1,2,3,4} f(k-t)+g(k-t+1, k)
$$

这里如果 $B_{i, j}$ 为一个合法字符 $g(i, j)=1$, 否则 $g(i, j)=0$ 。


## 4. WordPiece

WordPiece分词与BPE非常类似，只是在训缭阶段合并pair的策路不是pair的频率而是互信息。

$$
\operatorname{socre}=\log (p(a b))-(\log (p(a))+\log (p(b)))=\log (p(a b) / p(a) p(b))
$$

这里的动机是一个pair的频率很高，但是其中pair的一部分的频率更高，这时候不一定需要进行该pair的合并。而如果一个pair的频率很高，并且这个pair的两个部分都是只出现在这个pair中，就说明这个pair很值得合并。

- 训续方法: 从字符级的小词表出发，训绕产生合并规则以及一个词表
- 编码方法: 将文本切分成词，对每个词在词表中进行最大前向匹配
- 经典模型：BERT及其委列DistilBERT，MobileBERT等
