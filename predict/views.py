from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.urls import reverse
from .models import Predict

import keras
from keras.models import load_model

import re
seed = 13
import itertools
import numpy as np
import collections
np.random.seed(seed)
from itertools import chain
from sklearn.externals import joblib


def TransDict_from_list(groups):
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group) #Alphabetically sorted list
        for c in g_members:
            result[c] = str(tar_list[index]) #K:V map, use group's first letter as represent.
        index = index + 1
    return result


def translate_sequence (seq, TranslationDict):
#    import string
    from_list = []
    to_list = []
    for k,v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    return TRANS_seq


def get_composition(seq):
    aac=['G','A','V','L','I','F','W','Y','D','N','E','K','Q','M','S','T','C','P','H','R']
    length=len(seq)
    vector=[]
    
    '''
    four-parts composition 
    '''
    one_part=length//4
    dipeptide=[]
    chars = ['0', '1', '2', '3', '4', '5', '6']
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    iter = itertools.product(chars,repeat=2)
    for dip in list(iter):
        dipeptide.append(''.join(dip))
    #the first part vector        
    one_part_seq=seq[0:one_part]
    one_part_vector=[]
    for str in aac:
        one_part_vector.append(len(re.findall(r'(?='+str+')', one_part_seq))/one_part)
    one_part_seq = translate_sequence(one_part_seq, group_dict)
    for str in chars:
        one_part_vector.append(len(re.findall(r'(?='+str+')', one_part_seq))/one_part)
    for str in dipeptide:
        one_part_vector.append(len(re.findall(r'(?='+str+')', one_part_seq))/(one_part-1))
    vector.extend(one_part_vector)
    #the second part vector
    two_part_seq=seq[:one_part*2]
    two_part_vector=[]
    for str in aac:
        two_part_vector.append(len(re.findall(r'(?='+str+')', two_part_seq))/(one_part*2))
    two_part_seq = translate_sequence(two_part_seq, group_dict)
    for str in chars:
        two_part_vector.append(len(re.findall(r'(?='+str+')', two_part_seq))/(one_part*2))
    for str in dipeptide:
        two_part_vector.append(len(re.findall(r'(?='+str+')', two_part_seq))/(one_part*2 - 1))
    vector.extend(two_part_vector)
    #the third part vector
    three_part_seq=seq[:one_part*3]
    three_part_vector=[]
    for str in aac:
        three_part_vector.append(len(re.findall(r'(?='+str+')', three_part_seq))/(one_part*3))
    three_part_seq = translate_sequence(three_part_seq, group_dict)
    for str in chars:
        three_part_vector.append(len(re.findall(r'(?='+str+')', three_part_seq))/(one_part*3))
    for str in dipeptide:
        three_part_vector.append(len(re.findall(r'(?='+str+')', three_part_seq))/(one_part*3 - 1))
    vector.extend(three_part_vector)
    #the forth paty vector
    four_part_seq=seq
    four_part_vector=[]
    for str in aac:
        four_part_vector.append(len(re.findall(r'(?='+str+')', four_part_seq))/length)
    four_part_seq = translate_sequence(four_part_seq, group_dict)
    for str in chars:
        four_part_vector.append(len(re.findall(r'(?='+str+')', four_part_seq))/length)
    for str in dipeptide:
        four_part_vector.append(len(re.findall(r'(?='+str+')', four_part_seq))/(length-1))
    vector.extend(four_part_vector)
    
    return vector


def use_composition(sequence_dict):
    nueeric = []
    for k, v in sequence_dict.items():
        nueeric.append(get_composition(v))
    dataset=np.array(nueeric)
    return dataset


def read_fasta_text(fasta_text):
    seq_dict = collections.OrderedDict()
    name = ''
    fasta_list = fasta_text.split()
    names = []
    for line in fasta_list:
        line = line.rstrip()
        if line == '':
            continue
        if line[0] == '>':  # or line.startswith('>')
            line = line.split('|')
            name = line[0][1:].upper()  # discarding the initial >
            seq_dict[name] = ''
            names.append(name)
        elif name=='':
            break    
        else:
            seq_dict[name] = seq_dict[name] + line
    return seq_dict, names


def seq_predict(sequence_dict):
    com_fea = use_composition(sequence_dict)
    
    scaler = joblib.load('/home/DBP/model/MsDBN_scaler.h5')
    fea = scaler.transform(com_fea)
     
    keras.backend.clear_session()
    model = load_model('/home/DBP/model/MsDBN_model.h5')
    proba = model.predict(fea)
    return proba


def update_predict(request):

    #referer = request.META.get('HTTP_REFERER', reverse('results'))
    #数据检查
    text = request.POST.get('text', '').strip()
    if text == '':
        context = {}
        return render(request, 'error_empty.html', context)

    #检查通过，保存数据
    #predict = Predict()
    #predict.text = text
    #predict.save()

    #读数据,格式检查
    sequence_dict, seq_names = read_fasta_text(text)
    if len(sequence_dict) == 0:
        context = {}
        return render(request, 'error_format.html', context)

    #预测
    probas = seq_predict(sequence_dict)
    probas_one_D = list(chain(*probas))
    probas = np.array(probas)

    seq_names = np.array(seq_names)
    seq_names = seq_names.reshape(-1, 1)

    label = []
    for item in probas_one_D:
        if item > 0.5:
            label.append('Yes')
        else:
            label.append('No')
    label = np.array(label)
    labels = label.reshape(-1, 1)

    list_1 = np.concatenate((seq_names, probas), axis = 1)
    list_2 = np.concatenate((seq_names, labels), axis = 1)

    list_total = np.concatenate((list_1, list_2), axis = 0)
    datas = {}
    for item in list_total:
        pro_id = item[0]
        proba_label = item[1]
        datas.setdefault(pro_id, []).append(proba_label)

    return render(request, 'results.html', {'datas': datas})
