#coding: utf-8
import wave
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

CROP_MS     = 1000  #クロップする範囲(ms)
OVERRAP_MS  = 500   #クロップ時に重ねる範囲(ms)
DIVISION    = 150   #ペリオドグラムの時間分割数
PERIODOGRAM_SIZE = [28,28]  #ペリオドグラムのプーリング後サイズ


"""
    waveファイルを読み込む
"""
def _read_wave(file_name):
    try:
        wave_file = wave.open(file_name,"r") #Open
    except IOError as e:
        print("[ERROR] "+file_name+" is not found.")
        exit()
    return wave_file

"""
    音声区間識別のアルゴリズム
"""
#max mean
def _discrimination_algorithm_max_mean(whole_mean, part_max):
    return (part_max  > whole_mean)

#mean mean
def _discrimination_algorithm_mean_mean(whole_mean, part_mean):
    return (part_mean  > whole_mean)

def _discrimination_algorithm_therhold(part_mean):
    return (part_mean  > 127)


"""
    wavファイルからnumpy形式に変換する
    ステレオの場合は左のみ出力
     - file_name : 入力するファイルの名前
     return 変換した後のnumpy(一次元)
     return 音声ファイルのサンプリングレート
"""
def read(file_name):
    """読み込み作業"""
    wave_file = _read_wave(file_name)

    print("-----------------------------------------------------")
    print("FILE NAME            : ", file_name)
    #print("NONOLAL(1)/STEREO(2) : ", wave_file.getnchannels())  #モノラルorステレオ
    #print("BYTE 2(16bit)3(24bit): ", wave_file.getsampwidth())  #１サンプルあたりのバイト数
    #print("SAMPLING RATE(khz)   : ", wave_file.getframerate())  #サンプリング周波数
    #print("FLAME NUMBER         : ", wave_file.getnframes())    #フレームの総数

    """numpy形式に変換"""
    wave_flames  = wave_file.readframes(wave_file.getnframes())  #waveファイルのframeを読み込み
    wave_buffers = np.frombuffer(wave_flames, dtype= "int16")    #waveファイルをnumpy.arrayに変換
    print("WAVE BUFFER SIZE      : ", wave_buffers.shape)

    """一次元のnumpy形式で出力"""
    if wave_file.getnchannels() == 2:
        #サラウンド（2チャンネル）の場合
        l_buffers = wave_buffers[::wave_file.getnchannels()]
        r_buffers = wave_buffers[1::wave_file.getnchannels()]
        #DEBUG: 読み込んだwaveの全体波形
        #plt.plot(l_buffers)
        #plt.show()
        return l_buffers, wave_file.getframerate()
    return buffers, wave_file.getframerate()



"""
    音声numpyを、crop_msごとに切り取る
    - buffers : 入力されるnumpy形式の音声波形
    - sample_rate : サンプリング周波数
    return クロップされた音声波形（2次元）
"""
def crop(buffers, sample_rate):

    if(CROP_MS <= OVERRAP_MS):
        print("[NOTICE] overrap_ms is more than crop_ms")
        exit()

    """各クロップサイズ、オーバーラップ数をmsからサンプル数に変換"""
    crop_size       = int((sample_rate * CROP_MS) / 1000)       #切り取るサンプル数
    overrap_size    = int((sample_rate * OVERRAP_MS) / 1000)    #オーバーラップするサンプル数
    increase_size   = crop_size - overrap_size                  #繰り返しのときに進めるサンプル数
    iteral_max      = int(((buffers.shape[0] - crop_size) / (crop_size - overrap_size)) + 1) #buffersをはみ出ない繰り返し切り取り回数
    whole_mean      = np.mean(np.abs(buffers))                  #全体の平均

    cropped_buffers = []
    for i in range(iteral_max):
        cropped = buffers[((increase_size) * i):(((increase_size) * i) + crop_size)]    #クロップ
        part_mean   = np.mean(np.abs(cropped))       #区間の最大値

        #区間に音声が含まれているか識別
        if(not _discrimination_algorithm_therhold(part_mean)):
            continue
        cropped_buffers.append(cropped)

    return cropped_buffers

"""
    定区間からクロップされた音声波形を、分割時間のペリオドグラムにする
    - data : 音声データ波形
    return ペリオドグラム
"""
def periodogram(data):
    N = data.shape[0]  #データ全体のサンプル数
    Nd = int(N / DIVISION)  #分割時のサンプル数
    periodogram = np.zeros(int(int(Nd/2)*DIVISION))

    for t in range(DIVISION):
        data_p = data[(Nd * t):(Nd * (t + 1))] #分割数ごとにクロップ

        hanningWindow = np.hanning(Nd)              # ハニング窓
        dft = np.fft.fft(hanningWindow * data_p)    #離散フーリエ変換
        dft_abs = np.log(np.abs(dft) ** 2)          #振幅特性

        periodogram[(int(Nd/2) * t):((int(Nd/2) * (t+1)))] = dft_abs[:int(Nd/2)]

    periodogram = np.reshape(periodogram, (DIVISION,int(Nd/2)))
    ## DEBUG:
    #print("MAX",np.max(dataset_show))
    #plt.imshow(dataset_show, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.max(dataset_show)+1)
    #plt.show()

    #28*28にブーリング
    periodogram_pool = np.zeros(PERIODOGRAM_SIZE)
    a = int(periodogram.shape[0]/PERIODOGRAM_SIZE[0])
    b = int(periodogram.shape[1]/PERIODOGRAM_SIZE[1])
    for j in range(PERIODOGRAM_SIZE[1]):
        for i in range(PERIODOGRAM_SIZE[0]):
            periodogram_pool[j,i] = np.mean(periodogram[(j*a):(j*a+a),(i*b):(i*b+b)])

    #正規化
    periodogram_regu = np.floor( (periodogram_pool - np.min(periodogram_pool))* 127 / (np.max(periodogram_pool) - np.min(periodogram_pool)) )
    ## DEBUG:
    #print("MAX",np.max(dataset_d_pool))
    #plt.imshow(periodogram_regu, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.max(periodogram_regu))
    #plt.show()
    #exit()
    return periodogram_regu
