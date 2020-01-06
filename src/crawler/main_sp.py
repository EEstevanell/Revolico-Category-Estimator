from crawler.spider import *
import os
import pickle
import html2text
from scrapy.crawler import CrawlerProcess

def get_file_text(filename):
    with open(f'websites/{filename}', 'rb') as filehandle:
        url,filecontent = pickle.load(filehandle)
        # h = html2text.HTML2Text() #quedarme solo con el texto del html
        # h.ignore_images = True
        # h.ignore_links = True
        # text = h.handle(str(filecontent))
        return (url,filecontent) # 

def run_crawler(limit = 4000, seed = 'http://localhost:8000/Index-1.htm'):
    process = CrawlerProcess()
    # s = Spider()
    # s.limit = limit
    # s.start_urls = [seed]
    process.crawl(Spider, limit = limit, seed = seed)
    process.start()#iniciar el crawler
    # path = os.getcwd()+'/websites'


# for filename in os.listdir(path):
#     tup = get_file_text(filename)
#     print(tup[0])
#     print(tup[1])
#     print("")
#     print("")
  