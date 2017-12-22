#! /usr/bin/python
# -*- coding:utf-8 -*- 
import os
import pypinyin

def hanzi_to_pinyin(path_hanzi,path_pinyin):
    for file_txt in os.listdir(path_hanzi):
        if file_txt.find('.txt') >= 0:
            path_txt = os.path.join(path_hanzi,file_txt)
            file = open(path_txt)
            hanzi = file.read()
            file.close()
            print(hanzi)
            pinyin = pypinyin.pinyin(hanzi ,style=1,errors='default')
            file = open(path_pinyin,'a')
            print(pinyin)
            file.write('\n')
            file.write(file_txt)
            file.write(':  ')
            for str in pinyin:
                file.write(str[0])
                if str[0] != '\n' and str[0] != ' ' and str[0] != '\r' :file.write(' ')
                file.flush()
            file.close()
        

path_hanzi = 'C:\\Users\\wxin9\\Desktop\\hanzitopinyin\\hanzi'
path_pinyin = 'C:\\Users\\wxin9\\Desktop\\hanzitopinyin\\p.txt'
hanzi_to_pinyin(path_hanzi,path_pinyin)
















