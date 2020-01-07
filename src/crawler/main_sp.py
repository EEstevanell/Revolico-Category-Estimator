from crawler.spider import *
import os
import pickle
from scrapy.crawler import CrawlerProcess

def run_crawler(limit = 4000, seed = 'http://localhost:8000/Index-1.htm'):
    process = CrawlerProcess()
    process.crawl(Spider, limit = limit, seed = seed)
    process.start()#iniciar el crawler

