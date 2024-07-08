基于Shazam指纹的听音识曲,fingerprint压缩为32位整数存储
使用前先执行createbase.py创建音频数据,可选基于Mysql数据库存储数据或基于反向索引文件存储。
searchAudio.py为执行入口
其中有几个函数在代码中有多种实现。

**Audio Recognition Based on Shazam Fingerprinting**
Before using, execute `createbase.py` to create the audio data.
`searchAudio.py` serves as the entry point.
Several functions in the code have multiple implementations.