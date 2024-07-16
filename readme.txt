基于Shazam指纹的听音识曲,fingerprint压缩为32位整数存储
使用前先执行createbase.py创建音频数据,可选基于Mysql数据库存储数据或基于反向索引文件存储。
searchAudio.py为执行入口
其中有几个函数在代码中有多种实现。

**Audio Recognition Based on Shazam Fingerprinting**
Before using, execute `createbase.py` to create the audio data.
`searchAudio.py` serves as the entry point.
Several functions in the code have multiple implementations.

开发环境
编程语言：Python 3.11
主要音频处理库：librosa用于音频的加载、预处理和特征提取；scipy和numpy用于数据处理和数学运算；sounddevice用于麦克风收音。
数据库：
MySQL: 用于存储大规模音频指纹数据，进行持久化管理。
Pickle: 用于序列化对象，实现快速的数据存取，适用于较小规模的数据存储。(数据量小建议使用索引文件，数据量大建议使用数据库管理软件存储)
可视化：matplotlib用于绘制音频数据的频谱图和星座图，便于分析和调试。
开发工具：使用Vscode作为主要的开发环境支持代码编写、调试和版本管理。
版本控制：Git 2.45.2.windows.1，用于代码的版本控制。

1、音频数据处理与特征提取 (fingerprint.py):
该模块负责音频信号的加载、预处理、以及通过短时傅里叶变换（STFT）等方法提取音频的频谱特征。
从频谱图中提取关键点（即音频指纹），这些关键点通过计算得到的峰值点表示，用于后续的音频匹配和识别任务。
2、音频录制与输入处理 (record.py):
实现音频的实时录制功能，支持用户通过麦克风输入音频数据。
处理并转换录音数据为单声道，适配后续的特征提取和识别过程。
3、数据库操作与管理 (createdatabase.py, save_to_mysql.py, dorptable.py):
createdatabase.py 和 save_to_mysql.py 负责音频指纹的数据结构设计与音频数据的持久化存储。存储位置支持基于mysql的索引存储以及基于pickle文件的反向索引存储方式。
save_to_mysql.py使用MySQL数据库存储音频指纹，包括创建数据表、插入和更新记录。
dorptable.py 提供数据库表的清理功能，用于调试重新建表。
4、音频匹配与识别 (searchAudio.py):
利用存储在数据库中的音频指纹进行匹配，识别用户上传或录制的音频文件。
实现音频查询的处理逻辑，包括从数据库检索匹配的音频指纹，计算匹配度，输出最可能的匹配结果。

