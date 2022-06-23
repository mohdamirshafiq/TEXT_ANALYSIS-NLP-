# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:28:46 2022

@author: End User
"""
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

class model_training():
    def __init__(self):
        pass
    def plot_history_keys(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'],'r--',label='Training_loss')
        plt.plot(hist.history['val_loss'],label='validation_loss')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(hist.history['acc'],'r--',label='Training_acc')
        plt.plot(hist.history['val_acc'],label='validation_acc')
        plt.legend()
        plt.show()


class model_evaluation():
    def __init__(self):
        pass
    def evaluation_reports(self,y_true,y_pred):
        print('CLASSIFICATION_REPORT:\n',classification_report(y_true,y_pred))
        print('ACCURACY_SCORE: ',accuracy_score(y_true,y_pred))
        print('CONFUSION_MATRIX:\n',confusion_matrix(y_true,y_pred))