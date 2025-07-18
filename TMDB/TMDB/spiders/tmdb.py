# TMDB/TMDB/spiders/tmdb.py
import scrapy
from TMDB.items import TMDBItem
import pandas as pd

class TMDBSpider(scrapy.Spider):
    name = 'tmdb'
    allowed_domains = ['www.themoviedb.org']
    base_url = 'https://www.themoviedb.org/movie/'

    def start_requests(self):
        links_df = pd.read_csv(r"ml-32m\links.csv")
        tmdb_ids = links_df['tmdbId'].dropna().astype(int).tolist()
        batch_size = 1000
        for i in range(0, len(tmdb_ids), batch_size):
            batch_ids = tmdb_ids[i:i + batch_size]
            for tmdb_id in batch_ids:
                url = f"{self.base_url}{tmdb_id}"
                yield scrapy.Request(url, callback=self.parse, dont_filter=True)
                
    def parse(self, response):
        item = TMDBItem()
        item['tmdb_id'] = response.url.split('/')[-1]  # 提取tmdbId

        # 提取标题
        item['title'] = response.xpath('//*[@id="original_header"]/div[2]/section/div[1]/h2/a/text()').get().strip()

        # 提取简介
        description = response.xpath('//*[@id="original_header"]/div[2]/section/div[3]/div/p/text()').get()
        item['description'] = description.strip() if description else '暂无简介'

        # 提取导演（假设第一个导演）
        director = response.xpath('//*[@id="original_header"]/div[2]/section/div[3]/ol/li[1]/p[1]/a/text()').get()
        item['director'] = director.strip() if director else '未知'

        # 提取海报URL
        poster_url = response.xpath('//*[@id="original_header"]/div[1]/div[1]/div[1]/div/img/@src').get()
        item['poster_url'] = poster_url if poster_url else 'https://via.placeholder.com/300x450'

        yield item