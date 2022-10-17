from multiprocessing import Process
from column_model_ratewave import *

for i in range(10):
    p = Process(target=main, args=((i+5)*0.1*w_ex,i))  # target传入目标函数，args传入目标函数所需参数
    p.start()  # 启动进程