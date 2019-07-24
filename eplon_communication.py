# -*- coding: utf-8 -*-
# Main.py
from RN42 import RN42
from time import sleep
import RPi.GPIO as GPIO
import sys
import time


class Communication():
    ras = None
    
    def __init__(self):
        #""" Main Class """

        # Setting & Connect
        self.ras = RN42("ras", "44:85:00:EB:82:D1", 3)
        self.ras.connectBluetooth(self.ras.bdAddr, self.ras.port)

        print("Entering main loop now")

    def send(self, no):
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        try:
            print("BBBBBBBBBBBBBBBBBBBBBBBSS",no)
            self.ras.sock.send(str(no))
            print("data send:",no)
        except:
            import traceback
            traceback.print_exc()
            self.ras.disconnect(self.ras.sock)
            exit()
            
    def dis(self):
        self.ras.disConnect(self.ras.sock)
            
