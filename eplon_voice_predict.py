from keras import optimizers, losses
from keras.layers import *
from keras.models import Model, model_from_json
from keras.backend import int_shape
from keras.utils import to_categorical, plot_model
import numpy as np
import glob
import tqdm
import shutil
import os

DIR_PREDICT_OUTPUT = "./predict_441kHz_output"

#　入力ファイル名とラベル・ラベル番号付け
LABELSETS = [
    [0,"normal"],
    [1,"happy"],
]

#　予測データの読み込み
def input_predict_data(pathname):

    file_num = len(glob.glob(pathname+"/*"))
    print("total file numbers: ",file_num)

    data_X = np.zeros([file_num, 28, 28])

    for n in tqdm.tqdm(range(file_num)):
        data_X[n,:,:]   = np.load(pathname+"/"+str(n+1)+".npy")

    print("data_X shape:",data_X.shape)
    return data_X

#  予測の表示出力
def print_prediction(predict):
    i=0
    result = np.zeros(len(LABELSETS))
    for pred in predict:
        result[pred.argmax()] = result[pred.argmax()] + 1

    for pred in result:
        print(LABELSETS[i][1]+" : "+str(int(pred*1000/predict.shape[0])))
        i = i + 1
        
    return result.argmax()


#
def predict():

    model = model_from_json(open('net1.json').read())

    # 学習結果を読み込む
    model.load_weights('net1.h5')

    #print(model.summary())

    # Training configuration
    model.compile(loss=losses.categorical_crossentropy,
                    optimizer=optimizers.Adam(),
                    metrics=['accuracy'])


    X_test = input_predict_data(DIR_PREDICT_OUTPUT)

    # 配列の整形と，色の範囲を0-255 -> 0-1に変換
    X_test = X_test / 127
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # Evaluation
    result = model.predict(X_test, X_test.shape[0], 1)

    print("Prediction: ", result)
    ans = print_prediction(result)

    shutil.rmtree(DIR_PREDICT_OUTPUT)
    os.mkdir(DIR_PREDICT_OUTPUT)
    
    return ans
