import fasttext.FastText as fasttext
import torch
import re
from types import MethodType, FunctionType

import jieba

from collections import defaultdict

import  pandas  as pd

import  random
import  os

class TransformData(object):

    #将文档转换成 CSV
    def to_csv(self, inPath, outPath, index=False):
        dd = {}
        handler = open(inPath,encoding='utf-8')


        for line in handler:
            label, content = line.split(',', 1)
            key =label.strip('__label__').strip()
            if not  dd.get(key,False):
                dd[key] =[]
            dd[key].append(content.strip())
        handler.close()

        df = pd.DataFrame()
        for key in dd.keys():
            col = pd.Series(dd[key], name=key)
            df = pd.concat([df, col], axis=1)
        return df.to_csv(outPath, index=index, encoding='utf-8')
    #切割数据集 成 train.txt  test.txt
    def  SplitTrainTest(self,inPath,splitRate=0.8):
         baseName = inPath.rsplit('.',1)[0]
         trainFile = baseName + '_Train.txt'
         testFile = baseName+"_Test.txt"
         handle = pd.read_csv(inPath,index_col=False,low_memory=False)
         trainDataSet=[]
         testDataSet=[]
         for head  in list(handle.head()):
              print("head==",head,handle[head].dropna())
              trainNub= int(handle[head].dropna().__len__()*splitRate)
              subList=[f"__label__{head} , {item.strip()}\n" for item in handle[head].dropna().tolist()]
              trainDataSet.extend(subList[:trainNub])
              testDataSet.extend(subList[trainNub:])
              print("subList=",subList)

         random.shuffle(trainDataSet)
         random.shuffle(testDataSet)
         with open(trainFile, 'w', encoding='utf-8') as trainf, \
                 open(testFile, 'w', encoding='utf-8') as testf:
             for tmpItem in  trainDataSet:
                 trainf.write(tmpItem)
             for testItem in  testDataSet:
                 testf.write(testItem)


#########################





# 去除 字母 和字符
def ClearTxt(raw):
    fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
    return fil.sub(' ', raw)


# 去除停顿词
def StopWords(stopPath="./Data/stopwords.txt"):
    with open(stopPath, 'r', encoding='utf-8') as swf:
        return [line.strip() for line in swf]


# 用结巴分词分割变成 闲暇 友人 光顾 这种形式
def SegSentence(sentence, stopWord):
    sentence = ClearTxt(sentence)
    result = ' '.join([i for i in jieba.cut(sentence) if i.strip() and i not in stopWord])

    #print("seg==", result)
    return result





class UsingText:
    """
        训练一个监督模型, 返回一个模型对象

        @param input:           训练数据文件路径
        @param lr:              学习率
        @param dim:             向量维度
        @param ws:              cbow模型时使用
        @param epoch:           次数
        @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
        @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
        @param minn:            构造subword时最小char个数
        @param maxn:            构造subword时最大char个数
        @param neg:             负采样
        @param wordNgrams:      n-gram个数
        @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
        @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
        @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
        @param lrUpdateRate:    学习率更新
        @param t:               负采样阈值
        @param label:           类别前缀
        @param verbose:         ??
        @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
        @return model object
    """

    def __init__(self, inFilePath, dim=100, lr=0.1, epoch=150, loss="softmax", wordNgrams=2, prefixLabel="__label__"):
        self.ipt = inFilePath
        self.loss = loss
        self.wordNgrams = wordNgrams
        self.dim = dim
        self.lr = lr
        self.epoch = epoch
        self.modePath = "fasttext_model_all"#f"dim{str(self.dim)}_lr{str(self.lr)}_iter{str(self)}.model"
        self.prefixLable = prefixLabel

    # 开始训练
    def Train(self):
        if os.path.exists(self.modePath):
            self.classify = fasttext.load_model(self.modePath)
        else:
            self.classify = fasttext.train_supervised(self.ipt, \
                                                      label=self.prefixLable, dim=self.dim, \
                                                      epoch=self.epoch, lr=self.lr, \
                                                      wordNgrams=self.wordNgrams, \
                                                      )
            self.classify.save_model('fasttext_model_all.bin')

    def Test(self, testFilePath):
        result = self.classify.test_label(testFilePath)

        print("result==", result)

    # 计算精确度 召回率
    def CalPrecisionRecall(self, file='data_test.txt'):
        precision = defaultdict(int)
        recall = defaultdict(int)
        total = defaultdict(int)
        stopWord = StopWords()
        with open(file) as f:
            for line in f:
                label, content = line.split(',', 1)
                total[label.strip().replace(self.prefixLable,"")] += 1
                # labels2 = self.classify.predict([seg(sentence=content.strip(), sw='', apply=clean_txt)])

                contentList = [content.strip()]

                print("contentList==", contentList)

                labels2 = self.classify.predict(contentList)

                print("label2==", labels2)

                pre_label, sim = labels2[0][0][0], labels2[1][0][0]
                recall[pre_label.strip().replace(self.prefixLable,'')] += 1

                if label.strip() == pre_label.strip():
                    precision[label.strip().replace(self.prefixLable,'')] += 1

        print('precision', precision.keys())
        print('recall', recall.keys())
        print('total', total.keys())
        for sub in precision.keys():
            pre = precision[sub] / total[sub]
            rec = precision[sub] / recall[sub]
            F1 = (2 * pre * rec) / (pre + rec)
            print(f"{sub.replace(self.prefixLable,'')}  precision: {str(pre)}  recall: {str(rec)}  F1: {str(F1)}")



if __name__ == '__main__':

    # 分割
    if (not os.path.exists("../data/text/Out_Train.txt") or not os.path.exists('../data/text/Out_Test.txt')):
        transData = TransformData()

        transData.to_csv("../data/text/text.txt", "../data/text/Out.csv")

        transData.SplitTrainTest("../data/text/Out.csv")

    # 训练
    useFast = UsingText("../data/text/Out_Train.txt")
    useFast.Train()

    useFast.Test("../data/text/Out_Test.txt")
    # model=fasttext.load_model("fasttext_model_all.bin")
    # textvector=model.get_sentence_vector("hello world")
    # print(len(textvector),textvector)
    # print(type(textvector))
    # text=torch.from_numpy(textvector)
    # print(text)
    # text=torch.squeeze(text)
    # print(text)
    # 测试验证
    # useFast.CalPrecisionRecall("../data/text/Out_Test.txt")

    print("finish")
