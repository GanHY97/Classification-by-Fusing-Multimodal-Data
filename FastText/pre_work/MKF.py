import xlrd
import xlwt
import xlutils.copy as copy
import shutil
import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# from nltk.text import TextCollection
# import pandas as pd
# import gensim
# from numpy import array
# from numpy import argmax
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
#file='E:\YJS\medio\data\CrisisMMD_v2.0\annotations\hurricane_harvey_final_data_test.xlsx'
emoticons_str = r"""
(?:
[:=;] # 眼睛
[oO\-]? # ⿐鼻⼦子
[D\)\]\(\]/\\OpP] # 嘴
)"""
regex_str = [
emoticons_str,
r'<[^>]+>', # HTML tags
r'(?:@[\w_]+)', # @某⼈人
r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # 话题标签
r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
# URLs
r'(?:(?:\d+,?)+(?:\.?\d+)?)', # 数字
r"(?:[a-z][a-z'\-_]+[a-z])", # 含有 - 和 ‘ 的单词
r'(?:[\w_]+)', # 其他
r'(?:\S)' # 其他
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
def tokenize(s):
    return tokens_re.findall(s)
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def mkdict():
    #data= xlrd.open_workbook('E://YJS//medio//data//annotations//hurricane_harvey_final_data_test.xlsx')#文件名以及路径，如果路径或者文件名有中文给前面加一个r拜师原生字符。
    xlsxpath=r"E:\workspace\python\zywang_vgg\data\text_did"
    filelist=os.listdir(xlsxpath)
    fw = open("E:\\workspace\\python\\zywang_vgg\\data\\text\\" + "text_5_img.txt", 'w')  # 将要输出保存的文件地址
    for file in filelist:
        print(file)
        # fw = open("E:\\workspace\\python\\zywang_vgg\\data\\text\\"+"text.txt", 'w')  # 将要输出保存的文件地址
        #fw = open("E:\\workspace\\python\\zywang_vgg\\data\\text_do\\"+os.path.splitext(file)[0]+".txt", 'w')  # 将要输出保存的文件地址
        path="E:\\workspace\\python\\zywang_vgg\\data\\annotations\\"+file
        print(path)
        data = xlrd.open_workbook(path)
        table = data.sheet_by_index(0)



        nrows = table.nrows
        inf = []

        for x in range(1, nrows):
            # value = []
            # value.append(table.row_values(x, start_colx=2, end_colx=3)[0] == 'informative')
            cf = table.row_values(x, start_colx=0, end_colx=1)[0]
            if cf in inf:
                continue
            inf.append(cf)
            label = table.row_values(x, start_colx=4, end_colx=5)[0]
            a = table.row_values(x, start_colx=12, end_colx=13)[0]
            str1 = "__label__" + label + " , " #+table.row_values(x, start_colx=1, end_colx=2)[0]+" , "
            str = ""
            for i in range(len(a)):

                if (a[i] >= 'a' and a[i] <= 'z') or (a[i] >= 'A' and a[i] <= 'Z') or (a[i] >= '0' and a[i] <= '9') or a[
                    i] == ':' or a[i] == ' ' or a[i] == '@' or a[i] == '/':
                    str += a[i]
            # print(str)
            text = preprocess(str)
            # print("分词：", text)
            snowball_stemmer = SnowballStemmer("english")
            lancaster_stemmer = LancasterStemmer()
            porter_stemmer = PorterStemmer()
            text = [snowball_stemmer.stem(word) for word in text]
            # text=[lancaster_stemmer.stem(word) for word in text]
            # text=[porter_stemmer.stem(word) for word in text]
            #        print('Stemming:', text)
            wordnet_lemmatizer = WordNetLemmatizer()
            text = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in text]
            # print('Lemma:', text)
            text = [word for word in text if word not in stopwords.words('english')]
            # print('stopwords:', text)
            text = [word for word in text if
                    word != 'rt' and len(word) > 1 and word[:4] != 'http' and word[0] != ':' and word[0] != '@']
            # print(text)
            jishu = 0
            for word in text:
                if word[0] == '#':
                    text[jishu] = word[1:]
                str1 += word + ' '

                jishu += 1

            # print(a,text)
            # value.append(text)
            str1 = str1[:-1]
            #        inf[table.row_values(x, start_colx=0, end_colx=1)[0]] = value

            fw.write(str1.rstrip("\n"))  # 将字符串写入文件中
            # line.rstrip("\n")为去除行尾换行符
            fw.write("\n")  # 换行









if __name__ == '__main__':

    mkdict()
