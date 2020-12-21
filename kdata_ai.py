# encoding: UTF-8
# authoor email: szy@tsinghua.org.cn

# TradeEngineCnstk : 从tushare读取历史K线数据
# DeeplearningPredictStrategy : 预测策略，针对一只股票，使用它的历史K线训练模型，使当前时点到第predict_len个K线周期的收益率最大，最后测试模型
# DeeplearningRecognizeStrategy : 识别策略，训练模型，找到跟给定一批股票的K线走势相近的其他股票

import platform
import os
import sys
import time
import datetime
import pytz
import traceback
import math
import pandas

import tushare

#for deap learning
import numpy as np
import copy
import matplotlib.pyplot as plt
import keras as keras
import keras.datasets.imdb as imdb
import keras.datasets.reuters as reuters
import keras.preprocessing.sequence as sequence
import keras.models as models
import keras.layers as layers
import keras.regularizers as regularizers
from keras.callbacks import ModelCheckpoint
import multiprocessing


def get_savedata_dir():
    __SaveDataPathName__ = "sunquant"
    marketname = "kdata-ai"

    if platform.system() == "Windows":
        sd_path = os.path.join(os.getenv("appdata"), __SaveDataPathName__, marketname)
    else:
        sd_path = os.path.join(os.environ['HOME'], __SaveDataPathName__, marketname)
    if not os.path.exists(sd_path):
        os.makedirs(sd_path)
    return sd_path


class TradeEngineCnstk:

    def __init__(self):
        # variables which start with uppercase letter may have configuration in setting.json
        self.api = None

    @classmethod
    def load_kdata(cls, stockcode, start, end, ktype, load_from_file):
        datefmt = '%Y%m%d'
        if ktype in ['1min', '5min', '15min', '30min', '60min']:
            datefmt = '%Y-%m-%d %H:%M:%S'
        sd_path = get_savedata_dir()
        filepath = os.path.join(sd_path, stockcode + '-' + ktype + '.csv')
        if os.path.isfile(filepath):
            kdata_pd = pandas.read_csv(filepath, index_col=0, encoding='utf-8')
            tz = pytz.timezone('Etc/GMT-8')
            min_dt = datetime.datetime.now(tz)
            max_dt = datetime.datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tz)
            start_dt = max_dt if start is None else datetime.datetime.strptime(start, datefmt)
            end_dt = min_dt if end is None else datetime.datetime.strptime(end, datefmt)
            drop_indexes = []
            for dt_str, row in kdata_pd.iterrows():
                dt = datetime.datetime.strptime(str(dt_str), datefmt)
                if dt.timestamp() > max_dt.timestamp():
                    max_dt = dt
                if dt.timestamp() < min_dt.timestamp():
                    min_dt = dt
                if dt.timestamp() < start_dt.timestamp():
                    drop_indexes.append(dt_str)
                if dt.timestamp() > end_dt.timestamp():
                    drop_indexes.append(dt_str)
            if start_dt.month == 1 and start_dt.day == 1:
                start_dt = start_dt + datetime.timedelta(days=1)
            if start_dt.weekday() == 5:
                start_dt = start_dt + datetime.timedelta(days=2)
            if start_dt.weekday() == 6:
                start_dt = start_dt + datetime.timedelta(days=1)
            if end_dt.month == 1 and end_dt.day == 1:
                end_dt = end_dt - datetime.timedelta(days=1)
            if end_dt.weekday() == 5:
                end_dt = end_dt - datetime.timedelta(days=1)
            if end_dt.weekday() == 6:
                end_dt = end_dt - datetime.timedelta(days=2)
            print("min_dt="+str(min_dt))
            print("start_dt=" + str(start_dt))
            print("max_dt=" + str(max_dt))
            print("end_dt=" + str(end_dt))
            print("drop_indexes="+str(drop_indexes))
            if min_dt.timestamp() <= start_dt.timestamp() and max_dt.timestamp() >= end_dt.timestamp():
                kdata_pd.drop(index=drop_indexes, inplace=True)
                print("load_kdata, load from file:stockdode="+stockcode+",kdata_pd.len="+str(len(kdata_pd)))
                kdata = np.array(kdata_pd)
                return kdata

        if not load_from_file:
            kdata_pd = tushare.pro_bar(ts_code=stockcode, start_date=start, end_date=end, asset='E', adj='qfq', freq=ktype)
            # kdata_pd: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
            kdata_pd.drop(columns=['ts_code', 'pre_close', 'change', 'pct_chg', 'amount'], inplace=True)
            kdata_pd.to_csv(filepath, header=True, index=False, index_label='', encoding='utf-8')
            print("load_kdata, load from tushare:stockdode=" + stockcode + ",kdata_pd.len=" + str(len(kdata_pd)))
            kdata = np.array(kdata_pd)
            return kdata
        return None

    @classmethod
    def dotask_fetch_data(cls, stockcodes, ktype):
        sd_path = get_savedata_dir()

        if stockcodes is None:
            stockcodes = []
            all_stock_basics = tushare.pro_api().stock_basic(exchange='', list_status='L',
                                                             fields='ts_code,symbol,name,area,industry,list_date')
            for code, row in all_stock_basics.iterrows():
                stockcodes.append(row['ts_code'])

        print("dotask_fetch_data,stockcodes quantity=" + str(len(stockcodes)))
        print(stockcodes)

        i = 0
        for stockcode in stockcodes:
            try:
                i += 1
                print("No. " + str(i) + " : " + stockcode)
                filepath = os.path.join(sd_path, stockcode + '-' + ktype + '.csv')
                if not os.path.isfile(filepath):
                    TradeEngineCnstk.load_kdata(stockcode=stockcode, start=None, end=None, ktype=ktype,
                                                    load_from_file=False)
                    time.sleep(3)
            except Exception as e:
                print("dotask_fetch_data,Exception,e=" + str(e) + "traceback=\n" + traceback.format_exc())


def process_run_training_predict(stockcodes, ktype):
    for stockcode in stockcodes:
        print("\n"+stockcode)
        dl = DeeplearningPredictStrategy(stockcode=stockcode,
                                  epochs=30, batch_size=32, validation_split=0.2,
                                  train_block_dim=20, predict_len=5)

        #for day kdata
        train_time_start = '20180101'
        train_time_end = '20200630'
        test_time_start = '20200701'
        test_time_end = '20201130'

        #for 5 minutes kdata
        # train_time_start = '2020-11-16 09:35:00'
        # train_time_end = '2020-11-20 15:00:00'
        # test_time_start = '2020-11-23 09:35:00'
        # test_time_end = '2020-11-27 15:00:00'
        if dl.training_prepare(train_time_start, train_time_end, test_time_start, test_time_end, ktype, True):
            dl.training_run()
            # dl.training_showresult()
            # dl.simulate()
            time.sleep(2)

class DeeplearningPredictStrategy:

    def __init__(self, stockcode, epochs=20, batch_size=32, validation_split=0.2,
                 train_block_dim=20, predict_len=5):
        self._stockcode = stockcode

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.train_block_dim = train_block_dim
        self.predict_len = predict_len

        self.stock_features = 5         #stock_features：open, high, low, close, vol

    def generate_data_labels(self, kdata):
        ret_data = np.zeros((len(kdata), self.train_block_dim, self.stock_features), dtype=np.float32, order='C')
        ret_labels = np.zeros((len(kdata), 1), dtype=np.float32, order='C')

        blockcounter = 0

        mean = np.array(kdata.mean(axis=0))
        mean[0] = mean[1] = mean[2] = mean[3] = (sum(mean[0:4])/4.)
        kdata -= mean
        print("mean")
        print(str(mean))
        std = kdata.std(axis=0)
        has_zero = False
        for i in range(0, 4):
            has_zero = has_zero or std[i] == 0.
        if has_zero:
            std[0] = std[1] = std[2] = std[3] = 1.0
        if std[4] == 0:
            std[4] = 1.
        std[0] = std[1] = std[2] = std[3] = (sum(std[0:4])/4.)
        kdata /= std
        print("std")
        print(str(std))
        print("kdata,len="+str(len(kdata)))
        #print(str(kdata))
        for i in range(0, len(kdata) - self.train_block_dim - self.predict_len + 1):
            for j in range(self.train_block_dim):
                ret_data[blockcounter, self.train_block_dim-1-j] = kdata[i+self.predict_len+j, 0:5]
            X = np.array(range(0, self.predict_len+1), dtype=np.float32)
            mean = X.mean(axis=0)
            X -= mean
            std = X.std(axis=0)
            X /= std
            Y = kdata[i:i+self.predict_len+1, 0:4]
            Y = Y.mean(axis=1)
            Y = Y[::-1]
            p1 = np.polyfit(X, Y, 1)
            #print("X="+str(X)+",Y="+str(Y)+",p1="+str(p1))
            ret_labels[blockcounter] = 0.5 + 0.5*math.sin(math.atan(p1[0]))

            blockcounter += 1
            if blockcounter >= len(kdata):
                break

        ret_data = ret_data[:blockcounter]
        ret_labels = ret_labels[:blockcounter]
        print("generate_data_labels,ret_data.shape="+str(ret_data.shape))
        #print(str(ret_data))
        print("generate_data_labels,ret_labels.shape="+str(ret_labels.shape)+",std="+str(ret_labels.std(axis=0)))
        #print(str(ret_labels))
        return [ret_data, ret_labels]

    def training_prepare(self, train_time_start, train_time_end, test_time_start, test_time_end, ktype, load_from_file):
        kdata = TradeEngineCnstk.load_kdata(stockcode=self._stockcode, start=train_time_start, end=train_time_end, ktype=ktype,
                                load_from_file=load_from_file)
        if kdata is None:
            print("training_prepare,load_kdata ret None.stockcode="+self._stockcode)
            return False
        self.train_data, self.train_labels = self.generate_data_labels(kdata)

        kdata = TradeEngineCnstk.load_kdata(stockcode=self._stockcode, start=test_time_start, end=test_time_end, ktype=ktype,
                                load_from_file=load_from_file)
        if kdata is None:
            print("training_prepare,load_kdata ret None.stockcode="+self._stockcode)
            return False
        self.test_data, self.test_labels = self.generate_data_labels(kdata)
        return True

    def training_run(self, autosave=False):
        model = models.Sequential()
        model.add(layers.LSTM(1024, return_sequences=False, input_shape=(self.train_block_dim, self.stock_features)))
        #model.add(layers.LSTM(1024))
        model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        callbacks = None
        sd_path = get_savedata_dir()
        if autosave:
            checkpoint = ModelCheckpoint(filepath=os.path.join(sd_path, "weights.{epoch:02d}-{val_loss:.4f}.hdf5"),
                                         monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            callbacks = [checkpoint]

        self.history = model.fit(self.train_data, self.train_labels,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_split=self.validation_split,
                                 callbacks=callbacks)
        if autosave:
            model.save(filepath=os.path.join(sd_path, "weights_last.hdf5"))

        test_loss, test_metrics = model.evaluate(self.test_data, self.test_labels)
        print("training_run,test_loss:"+str(test_loss))
        print("training_run,test_mae:"+str(test_metrics))
        test_predict = model.predict(self.test_data)
        print("training_run,test_labels,len=" + str(len(self.test_labels)) + ",std=" + str(self.test_labels.std(axis=0)))
        print("training_run,test_predict,len=" + str(len(test_predict)) + ",std=" + str(test_predict.std(axis=0)))
        test_predict_diff = test_predict-self.test_labels
        print("training_run,test_predict_diff,len=" + str(len(test_predict_diff)) + ",std=" + str(test_predict_diff.std(axis=0)))

        sd_path = get_savedata_dir()
        filepath = os.path.join(sd_path, "training_result_predict.csv")
        already_exists = os.path.exists(filepath)
        with open(filepath, 'a', encoding='utf-8') as f:
            if not already_exists:
                f.write("stockcode,test_loss,test_mae,test_labels_len,test_labels_std,test_predict_len,test_predict_std,test_predict_diff_len,test_predict_diff_std\n")
            f.write(self._stockcode + ','
                    + str(round(test_loss, 6)) + ','
                    + str(round(test_metrics, 6)) + ','
                    + str(len(self.test_labels)) + ','
                    + str(round(self.test_labels.std(axis=0)[0], 6)) + ','
                    + str(len(test_predict)) + ','
                    + str(round(test_predict.std(axis=0)[0], 6)) + ','
                    + str(len(test_predict_diff)) + ','
                    + str(round(test_predict_diff.std(axis=0)[0], 6)) + "\n")
            f.close()

    def training_showresult(self):
        mae = self.history.history['mae']
        val_mae = self.history.history['val_mae']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(mae)+1)
        plt.plot(epochs, mae, 'bo', label='Training mae')
        plt.plot(epochs, val_mae, 'b', label='Validation mae')
        plt.title('Training and validation mae')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

    def simulate(self):
        sd_path = get_savedata_dir()
        model = models.load_model(os.path.join(sd_path, "weights_last.hdf5"))

        test_predict = model.predict(self.test_data)
        print("simulate,test_labels,len=" + str(len(self.test_labels)) + ",std=" + str(self.test_labels.std(axis=0)))
        print("simulate,test_predict,len=" + str(len(test_predict)) + ",std=" + str(test_predict.std(axis=0)))
        test_predict_diff = test_predict-self.test_labels
        print("simulate,test_predict_diff,len=" + str(len(test_predict_diff)) + ",std=" + str(test_predict_diff.std(axis=0)))

    @classmethod
    def dotask_training_all(cls, ktype):
        sd_path = get_savedata_dir()
        all_stockcodes = [f.split('-')[0] for f in os.listdir(sd_path) if f.endswith('-'+ktype+'.csv') and os.path.isfile(os.path.join(sd_path, f))]

        print("dotask_training_all,all_stockcodes quantity=" + str(len(all_stockcodes)))
        print(all_stockcodes)

        stockcode_cluster = []
        for i in range(len(all_stockcodes)):
            try:
                stockcode = all_stockcodes[i]
                stockcode_cluster.append(stockcode)
                if len(stockcode_cluster) >= 50 or i == len(all_stockcodes)-1:
                    print("\ndotask_training_all,stockcode_cluster,len="+str(len(stockcode_cluster)))
                    print(str(stockcode_cluster))
                    p = multiprocessing.Process(target=process_run_training_predict, args=(stockcode_cluster, ktype, ))
                    p.start()
                    p.join()
                    stockcode_cluster.clear()
                    time.sleep(300)
            except Exception as e:
                print("dotask_training_all,Exception,e=" + str(e) + "traceback=\n" + traceback.format_exc())


class DeeplearningRecognizeStrategy:

    def __init__(self, selected_stockcodes, epochs=20, batch_size=32, validation_split=0.2,
                 train_block_dim=20, predict_len=0):
        self._selected_stockcodes = selected_stockcodes

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.train_block_dim = train_block_dim
        self.predict_len = predict_len

        self.stock_features = 5         #stock_features：open, high, low, close, vol

        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None


    def generate_data_labels(self, stockcode, kdata):
        ret_data = np.zeros((len(kdata), self.train_block_dim, self.stock_features), dtype=np.float32, order='C')
        ret_labels = np.zeros((len(kdata), 1), dtype=np.float32, order='C')

        blockcounter = 0

        mean = np.array(kdata.mean(axis=0))
        mean[0] = mean[1] = mean[2] = mean[3] = (sum(mean[0:4])/4.)
        kdata -= mean
        print("mean")
        print(str(mean))
        std = kdata.std(axis=0)
        has_zero = False
        for i in range(0, 4):
            has_zero = has_zero or std[i] == 0.
        if has_zero:
            std[0] = std[1] = std[2] = std[3] = 1.0
        if std[4] == 0:
            std[4] = 1.
        std[0] = std[1] = std[2] = std[3] = (sum(std[0:4])/4.)
        kdata /= std
        print("std")
        print(str(std))
        print("kdata,len="+str(len(kdata)))
        #print(str(kdata))
        for i in range(0, len(kdata) - self.train_block_dim - self.predict_len + 1):
            for j in range(self.train_block_dim):
                ret_data[blockcounter, self.train_block_dim-1-j] = kdata[i+self.predict_len+j, 0:5]
            ret_labels[blockcounter] = 1 if stockcode in self._selected_stockcodes else 0

            blockcounter += 1
            if blockcounter >= len(kdata):
                break

        ret_data = ret_data[:blockcounter]
        ret_labels = ret_labels[:blockcounter]
        print("generate_data_labels,ret_data.shape="+str(ret_data.shape))
        #print(str(ret_data))
        print("generate_data_labels,ret_labels.shape="+str(ret_labels.shape)+",std="+str(ret_labels.std(axis=0)))
        #print(str(ret_labels))
        return [ret_data, ret_labels]

    def training_prepare_onestock(self, stockcode, train_time_start, train_time_end, test_time_start, test_time_end, ktype, load_from_file):
        kdata = TradeEngineCnstk.load_kdata(stockcode=stockcode, start=train_time_start, end=train_time_end, ktype=ktype,
                                load_from_file=load_from_file)
        if kdata is None:
            print("training_prepare_onestock,load_kdata ret None.stockcode="+stockcode)
            return False
        train_data, train_labels = self.generate_data_labels(stockcode, kdata)
        if self.train_data is None:
            self.train_data = train_data
            self.train_labels = train_labels
        else:
            self.train_data = np.append(self.train_data, train_data, axis=0)
            self.train_labels = np.append(self.train_labels, train_labels, axis=0)

        kdata = TradeEngineCnstk.load_kdata(stockcode=stockcode, start=test_time_start, end=test_time_end, ktype=ktype,
                                load_from_file=load_from_file)
        if kdata is None:
            print("training_prepare_onestock,load_kdata ret None.stockcode="+stockcode)
            return False
        test_data, test_labels = self.generate_data_labels(stockcode, kdata)
        if self.test_data is None:
            self.test_data = test_data
            self.test_labels = test_labels
        else:
            self.test_data = np.append(self.test_data, test_data, axis=0)
            self.test_labels = np.append(self.test_labels, test_labels, axis=0)

        print("training_prepare_onestock,stockcode="+stockcode
              +" train_data.shape="+str(self.train_data.shape)+" train_labels.shape="+str(self.train_labels.shape)
              +" test_data.shape="+str(self.test_data.shape)+" test_labels.shape="+str(self.test_labels.shape))
        return True

    def training_prepare_allstocks(self, ktype):
        sd_path = get_savedata_dir()
        all_stockcodes = [f.split('-')[0] for f in os.listdir(sd_path) if f.endswith('-'+ktype+'.csv') and os.path.isfile(os.path.join(sd_path, f))]

        print("training_prepare_allstocks,all_stockcodes quantity=" + str(len(all_stockcodes)))
        print(all_stockcodes)


        #for day kdata
        train_time_start = '20180101'
        train_time_end = '20200630'
        test_time_start = '20200701'
        test_time_end = '20201130'

        #for 5 minutes kdata
        # train_time_start = '2020-11-16 09:35:00'
        # train_time_end = '2020-11-20 15:00:00'
        # test_time_start = '2020-11-23 09:35:00'
        # test_time_end = '2020-11-27 15:00:00'
        for stockcode in all_stockcodes:
            print("\n"+stockcode)
            self.training_prepare_onestock(stockcode, train_time_start=train_time_start, train_time_end=train_time_end,
                                           test_time_start=test_time_start, test_time_end=test_time_end,
                                           ktype=ktype, load_from_file=True)
        return True

    def training_run(self, autosave=False):
        model = models.Sequential()
        model.add(layers.LSTM(1024, return_sequences=False, input_shape=(self.train_block_dim, self.stock_features)))
        #model.add(layers.LSTM(1024))
        model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = None
        sd_path = get_savedata_dir()
        if autosave:
            checkpoint = ModelCheckpoint(filepath=os.path.join(sd_path, "weights.{epoch:02d}-{val_loss:.4f}.hdf5"),
                                         monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            callbacks = [checkpoint]

        self.history = model.fit(self.train_data, self.train_labels,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_split=self.validation_split,
                                 callbacks=callbacks)
        if autosave:
            model.save(filepath=os.path.join(sd_path, "weights_last.hdf5"))

        test_loss, test_metrics = model.evaluate(self.test_data, self.test_labels)
        print("training_run,test_loss:"+str(test_loss))
        print("training_run,test_metrics:"+str(test_metrics))
        test_predict = model.predict(self.test_data)
        print("training_run,test_labels,len=" + str(len(self.test_labels)) + ",std=" + str(self.test_labels.std(axis=0)))
        print("training_run,test_predict,len=" + str(len(test_predict)) + ",std=" + str(test_predict.std(axis=0)))
        test_predict_diff = test_predict-self.test_labels
        print("training_run,test_predict_diff,len=" + str(len(test_predict_diff)) + ",std=" + str(test_predict_diff.std(axis=0)))

        sd_path = get_savedata_dir()
        filepath = os.path.join(sd_path, "training_result_recognize.csv")
        already_exists = os.path.exists(filepath)
        with open(filepath, 'a', encoding='utf-8') as f:
            if not already_exists:
                f.write("selected_stocks,test_loss,test_mae,test_labels_len,test_labels_std,test_predict_len,test_predict_std,test_predict_diff_len,test_predict_diff_std\n")
            f.write(str(self._selected_stockcodes) + ','
                    + str(round(test_loss, 6)) + ','
                    + str(round(test_metrics, 6)) + ','
                    + str(len(self.test_labels)) + ','
                    + str(round(self.test_labels.std(axis=0)[0], 6)) + ','
                    + str(len(test_predict)) + ','
                    + str(round(test_predict.std(axis=0)[0], 6)) + ','
                    + str(len(test_predict_diff)) + ','
                    + str(round(test_predict_diff.std(axis=0)[0], 6)) + "\n")
            f.close()

    def training_showresult(self):
        metrics = self.history.history['accuracy']
        val_metrics = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(metrics)+1)
        plt.plot(epochs, metrics, 'bo', label='Training metrics')
        plt.plot(epochs, val_metrics, 'b', label='Validation metrics')
        plt.title('Training and validation mae')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

    def simulate(self):
        sd_path = get_savedata_dir()
        model = models.load_model(os.path.join(sd_path, "weights_last.hdf5"))

        test_predict = model.predict(self.test_data)
        print("simulate,test_labels,len=" + str(len(self.test_labels)) + ",std=" + str(self.test_labels.std(axis=0)))
        print("simulate,test_predict,len=" + str(len(test_predict)) + ",std=" + str(test_predict.std(axis=0)))
        test_predict_diff = test_predict-self.test_labels
        print("simulate,test_predict_diff,len=" + str(len(test_predict_diff)) + ",std=" + str(test_predict_diff.std(axis=0)))

    @classmethod
    def dotask_training_all(cls, selected_stockcodes, ktype):
        print("\n"+str(selected_stockcodes))
        dl = DeeplearningRecognizeStrategy(selected_stockcodes=selected_stockcodes,
                                  epochs=30, batch_size=128, validation_split=0.2,
                                  train_block_dim=60, predict_len=0)

        if dl.training_prepare_allstocks(ktype=ktype):
            print(" train_data.shape=" + str(dl.train_data.shape) + " train_labels.shape=" + str(dl.train_labels.shape)
                  +" test_data.shape=" + str(dl.test_data.shape) + " test_labels.shape=" + str(dl.test_labels.shape))
            dl.training_run()
            dl.training_showresult()
            # dl.simulate()


if __name__ == '__main__':
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)

    tushare.set_token(token='1a312d7a80c6fcc1fd0a28116f8a1988b6756189d4db76e5bc603031')

    TradeEngineCnstk.dotask_fetch_data(stockcodes=['000001.SZ', '600036.SH', '601318.SH'], ktype='D')
    DeeplearningPredictStrategy.dotask_training_all(ktype='D')

    #TradeEngineCnstk.dotask_fetch_data(stockcodes=None, ktype='D')
    #sss = ['002271.SZ', '603713.SH', '603208.SH', '001914.SZ', '603605.SH', '002511.SZ',
    #       '002959.SZ', '002705.SZ', '603288.SH', '603027.SH', '002847.SZ', '600882.SH',
    #       '603345.SH', '002557.SZ', '600519.SH', '000858.SZ', '600276.SH', '300558.SZ',
    #       '300601.SZ', '300529.SZ', '600763.SH', '300015.SZ', '300347.SZ', '300759.SZ',
    #       '603127.SH', '300760.SZ', '002901.SZ', '002821.SZ', '603939.SH', '603233.SH',
    #       '600436.SH', '603517.SH', '600298.SH', '600809.SH', '002475.SZ', '300122.SZ']
    #DeeplearningRecognizeStrategy.dotask_training_all(selected_stockcodes=sss, ktype='D')
    exit(0)

