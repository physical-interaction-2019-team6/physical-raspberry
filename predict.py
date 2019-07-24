import eplon_voice_generate
import eplon_voice_predict
import eplon_rec
import eplon_communication

com = eplon_communication.Communication()

while True:
    try:
        print("REC")
        eplon_rec.rec()
        
        print("GEBERATE")
        if(True == eplon_voice_generate.generate_predict_data()):
            
            print("PREDICT")
            result = eplon_voice_predict.predict()
        else:
            
            result = 0
            
        print("#########################")
        print("ANS: ",result)
        
        com.send(result)
    except:
        com.dis()


