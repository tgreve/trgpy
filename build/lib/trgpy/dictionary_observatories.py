from collections import OrderedDict
# *******************************************************
# band[]:
#     Recording frequency range For each receiver band
#     for a number of observatories.
#     freq[H2OJlo]=frequency[GHz]
# *******************************************************
band=OrderedDict()
band['ALMA']=[['Band 3',84.  ,116.],['Band 4',125. ,163.],
              ['Band 5',163. ,211.],['Band 6',211. ,275.],
              ['Band 7',275. ,373.],['Band 8',385. ,500.],
              ['Band 9',602. ,720.],['Band 10',787. ,950.]]            #[GHz]
#band['SMA']=[['Band 1',186.,242.],['Band 2',271.5,349.5],
#             ['Band 3',330.,420.]]                                     #[GHz]
band['SMA']=[['Band 1',194.,240.],['Band 2',210.,270.],
             ['Band 3',258.,351.], ['Band 4',336.,408.]]                                     #[GHz]
band['IRAM-30m']=[['EMIR E0',73.  ,117.],['EMIR E1',125. ,184.],
                  ['EMIR E2',202. ,274.],['EMIR E3',277. ,350.],
                  ['HERA1'  ,215. ,272.],['HERA2'  ,215. ,241.]]       #[GHz]
band['IRAM NOEMA']=[['Band 1',80.  ,116.],['Band 2',129. ,177.],
                    ['Band 3',201.,267.],['Band 4',277. ,371.]]        #[GHz]
band['JVLA']=[['B4',0.058  ,0.084],['90cm (P)',0.23,0.47],['20cm (L)',1.0,2.0],
              ['13cm (S)',2.0,4.0],['6cm (C)',4.0,8.0]   ,['3cm (X)',8.0,12.0],
              ['2cm (Ku)',12.0,18.0],['1.3cm (K)',18.0,26.5],['1cm (Ka)',26.5,40.0],
              ['0.7cm (Q)',40.0,50.0]]                                         #[GHz]
band['FIRI']=[['Band 1',749.481145 ,11991.69832]]                      #[GHz]
band['JCMT']=[['RxA',212.,274.],['HARP',325.,375],['RxW',620,710]]                #[GHz]
band['GBT']=[['PF1',0.290,0.920],['PF2',0.910,1.23],['L',1.15,1.73],
            ['S',1.73,2.60],['C',3.95,7.8],['KFPA',18.0,27.5],
            ['Ka',26.0,39.5],['Q',39.2,50.5],['Ku',12.0,15.4],
            ['X',7.80,11.6],['W',67,93],['ARGUS',74,116]] #[GHz]
