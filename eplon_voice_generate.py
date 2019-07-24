#coding: utf-8
import eplon_voice_preprocess
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm


#入力ファイル名とラベル・ラベル番号付け
LABELSETS = [
    [0,"normal"],
    [1,"happy"],
]

"""
inputのファイル構造は以下のようにする
    [dataset]
    - [tension]
    - - aaa.wav
    - - bbb.wav
    - [frustrated]
    - - ccc.wav
    - [fun]
    ...
"""
DIR_DATASET_INPUT = "./dataset_441kHz"
DIR_DATASET_OUTPUT = "./dataset_441kHz_output"
DIR_PREDICT_INPUT = "./predict_441kHz"
DIR_PREDICT_OUTPUT = "./predict_441kHz_output"


#読み込む学習ファイルが全てあるか確認
def _exist_input_learningfiles():
    for labelset in LABELSETS:
        if(len(glob.glob(DIR_DATASET_INPUT+"/"+labelset[1]+"/*.wav")) < 1):
            print("[ERROR] "+DIR_DATASET_INPUT+"/"+labelset[1]+"/ is not found or no file.")
            exit()

#読み込む予測ファイルが全てあるか確認
def _exist_input_predictfiles():
    if(not len(glob.glob(DIR_PREDICT_INPUT+"/*.wav")) == 1):
        print("[ERROR] "+DIR_PREDICT_INPUT+"/ must have only one wav file.")
        exit()

#書き込む学習ファイルが存在しているか確認
def _exist_output_learningfolder():
    if(not os.path.exists(DIR_DATASET_OUTPUT)):
        print("[ERROR] "+DIR_DATASET_OUTPUT+" is not found.")
        exit()

"""
学習データを読み込む
"""
def generate_learning_data():

    _exist_input_learningfiles()  #入力フォルダが存在しているか
    _exist_output_learningfolder()  #入力フォルダが存在しているか

    i = 0
    for labelset in LABELSETS:
        for filepath in glob.glob(DIR_DATASET_INPUT+"/"+labelset[1]+"/*.wav"):
            #[1] wav音声波形ファイルをnumpy形式へ
            buffers, sample_rate    = eplon_voice_preprocess.read(filepath)
            #[2] numpy音声波形を短い間隔で切り取る
            cropped_buffers         = eplon_voice_preprocess.crop(buffers, sample_rate)

            for cropped_buffer in tqdm(cropped_buffers):
                i = i + 1
                #[3] 切り取られたnumpy音声波形を、一定時間ごとにDFT、ペリオドグラムを生成
                periodogram         = eplon_voice_preprocess.periodogram(cropped_buffer)
                #[4] ラベルデータとペリオドグラムデータをnumpy形式で保存
                np.save(DIR_DATASET_OUTPUT+"/data/"+str(i)+".npy",  periodogram)
                np.save(DIR_DATASET_OUTPUT+"/label/"+str(i)+".npy", labelset[0])
    print("SUCCESS FULLY!!")

"""
予測したいデータを読み込む
"""
def generate_predict_data():

    _exist_input_predictfiles()

    #[1] wav音声波形ファイルをnumpy形式へ
    filenames = glob.glob(DIR_PREDICT_INPUT+"/*.wav")
    buffers, sample_rate    = eplon_voice_preprocess.read(filenames[0])
    #[2] numpy音声波形を短い間隔で切り取る
    cropped_buffers         = eplon_voice_preprocess.crop(buffers, sample_rate)
    
    if(len(cropped_buffers)==0):
        return False

    i = 0
    for cropped_buffer in tqdm(cropped_buffers):
        i = i + 1
        #[3] 切り取られたnumpy音声波形を、一定時間ごとにDFT、ペリオドグラムを生成
        periodogram         = eplon_voice_preprocess.periodogram(cropped_buffer)
        #[4] ラベルデータとペリオドグラムデータをnumpy形式で保存
        np.save(DIR_PREDICT_OUTPUT+"/"+str(i)+".npy",  periodogram)
    return True
