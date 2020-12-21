# kdata_ai
Deep learning and calculate the historical K-line data and predict the future trend.
The historical K-line data used in this program is from Chinese Stock Market.

Author email: szy@tsinghua.org.cn

人工智能根据历史K数预测未来走势的计算

（一）结论

    (1). 仅靠历史日K线，无法预测未来几天的走势。
    (2). 现价就是最好的价格。
        仅从日K线上看的话，现价是一个已经包含了对未来各种可能进行平衡的一个价格，它就是最好的价格。
        换句话说，你仅根据日K线，以现价买入一只股票，既没有便宜可占，也没有吃亏，即使今天的价格和昨天相比，有很大
    的不同，它也只是对最近一天内的事件的重新平衡，以达到一个新的最合适的价格。这跟奥斯本的股价“随机漫步”理论吻合，
    也与尤金法玛的“有效市场假说”中的弱式有效很吻合。《Python深度学习》一书中也提到，股票的历史走势数据不包含未来的
    信息。
        题外话，A股市场目前还仅处于弱式有效阶段，尚未达到“半强式有效”，也就是通过股价历史走势之外的信息分析，
    比如基本面信息，事件影响，是可以比较容易取得超越指数的收益的。
    (3). 在人工智能的神经网络中，已做到最大限度的对m个日K线所含未来走势信息的提取，这个提取强度，要远远大于传统技术指
    标，比如RSI、KDJ、MACD等对未来信息的提取。此算法不能取得超额收益，意味着传统技术指标也无法取得超额收益。
    (4). 使用更高频的K线，比如5分钟线，1分钟线，比使用日K线的效果略好一些，能否达到可以赚钱的程度，尚需验证。
    (5). 现有的一些量化基金是怎么取得超额收益的？
        量化基金使用了更高频的数据（分笔交易数据），以及股票基本面的因子，或者其他一些事件因子。

（二）本程序算法架构
    1. 训练模型：已知m+n个日K线数据，给定前m日的K线数据，标签记为接下来n日内的股价走势斜率。
    2. 评估模型：给定前m个日K线数据，预测未来n日内的股价走势斜率，以找到最可能上涨或下跌的股票，计算结果表明，未来
    的斜率趋近于0。

（三）人工智能开发环境安装

    参考《Python深度学习》一书，作者 弗朗索瓦.肖莱（Francois Chollet），张亮译

    (1).硬件
    
        1.  GPU，本程序的计算量不大，不需要单独购买GPU。
            如果想使用GPU做AI计算，推荐购买英伟达 RTX 2080Ti。
    
    (2).安装
    
        1. jupyter 笔记本开发台（非必需）
            pip install jupyterlab
            pip install notebook
            jupyter notebook
        2. 科学套件（Anaconda已默认安装）
            OpenBLAS，在CPU上运行快速张量运算，openblas.net（非必需）
            Numpy, sciPy Matplotlib
            HDF5，（h5py）， 高效保存数值数据大文件
        3. GPU支持，nvidia.com（如果已有英伟达GPU硬件）
            CUDA：https://developer.nvidia.com/cuda-downloads
            cuDNN：https://developer.nvidia.com/cudnn
            下载后的cuDNN文件复制到CUDA对应目录下即可。
        4. 安装TensorFlow
            pip install tensorflow
            pip install tensorflow-gpu （如果有GPU硬件和开发套件）
        5. 安装Keras
            Graphviz 和 pydot-ng， 可视化Keras模型
            pip install keras
            源码：git clone https://github.com/fchollet/keras
    
    (3).运行
    
        1.  运行示例：		python examples/mnist_cnn.py
             自动产生配置文件：	~/.keras/keras.json
             Ubuntu系统监视CPU：	$ watch -n 5 NVIDIA-smi -a --display=utilization
             《Python深度学习》代码示例：	https://www.manning.com/books/deep-learning-with-python
                        https://github.com/fchollet/deep-learning-with-python-notebooks
    
        2. 库
            from keras.datasets import imdb
            from keras.preprocessing import sequence
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Embedding, SimpleRNN, LSTM
            from keras import regularizers
            import matplotlib.pyplot as plt
