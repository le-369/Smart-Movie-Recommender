# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class TMDBItem(scrapy.Item):
    tmdb_id = scrapy.Field()
    title = scrapy.Field()
    description = scrapy.Field()
    director = scrapy.Field()
    poster_url = scrapy.Field()
