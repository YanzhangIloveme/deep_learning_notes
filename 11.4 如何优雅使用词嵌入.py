# 以腾讯词嵌入为例，腾讯中文的词向量映射集在解压后有15.5G，共计有8,824,330条字词短语，内存较小的计算机显然不能直接加载，故为满足小内存、低性能的计算机的需要，
# 特建立对词嵌入的映射关系文件，映射后只有313MB，满足了此类计算机的需求。

# 原理是什么？
# 这里是典型的以时间换空间的方式解决在使用腾讯词嵌入的时候内存资源耗尽的问题。在词嵌入与程序之间建立一个中间的映射文件，程序通过映射文件读取词嵌入的内容，
# 程序通过词汇可以访问到对应的文件指针的起始位置以及读取长度，然后程序就可以直接访问磁盘中的对应的数据了。映射文件格式如下：
# 1 词汇名称
# 2 索引
# 3 起始 （文件指针起始位置）
# 4 长度 （文件指针读取长度）

#%% How?

# 1 安装linecache:随机读写文本行
"""
linecache 模块允许从一个 Python 源文件中获取任意的行，并会尝试使用缓存进行内部优化，常应用于从单个文件读取多行的场合。
此模块被 traceback 模块用来提取源码行以便包含在格式化的回溯中。
"""
# import linecache

# 2 创建文件夹
#   创建名字叫做“utils”的文件夹，里面放入“ShowProcess .py”，这个可以在文章末尾复制代码，可以直接从文中的百度网盘链接中下载。
#   创建名字叫做“embeddings”的文件夹，里面放入解压好了的“Tencent_AILab_ChineseEmbedding.txt”文件，以及“ReadEmbeddings .py”，
#   映射文件“embeddings_map_index.txt”也将在这里生成。

# 3 加载模块
# from embeddings.ReadEmbeddings import ReadEmbeddings
# EMB = ReadEmbeddings()

# 4 关键参数设置（可选）：指定词嵌入文件位置（默认使用腾讯词嵌入）。
# EMB.emb_file="要指定的词嵌入文件的位置，默认设置：embeddings/Tencent_AILab_ChineseEmbedding.txt"
# 指定词嵌入词条数量（默认使用腾讯词嵌入的词条数量）。
# EMB.max_count = 8824330
# 如果有需要指定生成的映射文件的位置，可以在这里指定。
# EMB.map_file="要指定的生成映射文件的位置，默认设置：embeddings/embeddings_map_index.txt"

# 5（首次使用）创建映射文件   这个过程需要一个小时左右，可以选择自己生成，也可以选择博主生成好了的文件。
# EMB.creat_map_file()

# 6 加载词嵌入映射文件
#   生成完之后就可将映射文件加载进内存了，你可自行查看映射列表的内容。
# map_list = []
# map_list = EMB.load_map_in_memery()

# 7 单个查询
# word = '你好'
# value = []
# value = EMB.find_by_word(map_list, word)

# 8 批量查询
# query_list = ['这个','世界','需要','更多的','英雄']
# return_dict = {}
# return_dict = EMB.find_by_list(map_list, query_list)

# 9 释放内存：  当不再需要映射文件时，立即释放内存。
# EMB.clear_cache()

#%% 高级用法
# 1 指定编码 :   在初始化的时候就可指定全局编码，在读取词嵌入以及创建映射文件的时候可以使用统一的编码。
# EMB.encoding = '指定文件编码，默认设置：utf8'

# 2 启用日志 :   单独查询与批量查询都具备写入日志的功能(仅记录查询失败日志)。
"""
#--------------------------------方法一--------------------------------

log_obj = open("log_file.log", "a+",encoding=EMB.encoding) #或者自行指定encoding
# 单独查询
word = "你好"
value = EMB.find_by_word(map_list, word, f_log_obj=log_obj)

# 批量查询
query_list = ['这个','世界','需要','更多的','英雄']
return_dict = {}
return_dict = EMB.find_by_list(map_list, query_list, f_log_obj=log_obj)

# 当不再需要记录查询日志时
log_obj.close()

#--------------------------------方法二--------------------------------
with open("log_file.log", "a+", encoding=EMB.encoding) as log_obj:
    ...

"""

# 3 启用元素删除功能
#   元素删除功能在每个词汇仅查询一次的条件下才能启用，此功能在需要查询的词汇量特别大的时候会显著提升查询效率，有效减少查询时间，举个栗子：

"""
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset_all.tolist())
vocab = tokenizer.word_index
# 比如： vocab["你好"] == 10
# 那么“你好”的排名就是10
# 比如你发现len(vocab) == 1000000
# 你就需要创建一个(1,000,001, 200)的矩阵,200这个参数是依据你词嵌入的维度来决定的。
# 你可以翻转key与word构成vocab_resverse，然后使用下面的语句

matrix[10] = EMB.find_by_word(map_list, vocab_resverse[10], is_del_element=True)
由于Python的特性，此功能会影响到外部变量，受影响的外部变量：

map_list = EMB.load_map_in_memery()
A = map_list
B = A

# 这里A、B、map_list均会受到元素删除功能的影响
word = "你好"
value = EMB.find_by_word(map_list, word, is_del_element=True)
# 这里“你好”一词已经从“map_list”中移除，无法再被查询到。

#虽然已经明确说了每个词汇仅查询一次的条件，但是还是给出解决办法：
#--------------------------------方法一--------------------------------
import copy
map_list = EMB.load_map_in_memery()
A = copy.deepcopy(map_list)

# 这里仅A会受到元素删除功能的影响
word = "你好"
value = EMB.find_by_word(A, word, is_del_element=True)
# 这里“你好”一词已经从“A”中移除，无法再被A查询到，但是map_list不受影响。

#--------------------------------方法二--------------------------------
# 重新加载词嵌入映射文件
map_list = EMB.load_map_in_memery()
"""
