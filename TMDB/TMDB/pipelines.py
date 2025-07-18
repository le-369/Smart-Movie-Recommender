# TMDB/TMDB/pipelines.py
import json
import os

class TMDBPipeline:
    def open_spider(self, spider):
        self.file = open(r'TMDB\movies_tmdb.json', 'w', encoding='utf-8')  # 以写入模式打开，覆盖旧文件
        self.file.write('[')  # 写入开括号
        self.first_item = True  # 标记是否为第一个item

    def process_item(self, item, spider):
        # 转换为字典并序列化
        item_dict = dict(item)
        item_json = json.dumps(item_dict, ensure_ascii=False)
        
        # 第一个item不加逗号，其他item前加逗号
        if not self.first_item:
            self.file.write(',\n')
        self.file.write(item_json)
        self.file.flush()  # 立即写入磁盘
        self.first_item = False
        return item

    def close_spider(self, spider):
        self.file.write(']')  # 写入闭括号
        self.file.close()