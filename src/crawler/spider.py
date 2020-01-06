import scrapy
import html2text
import pickle

from scrapy.linkextractors import LinkExtractor

class Spider(scrapy.Spider):
    name = "doc"
    start_urls = ['http://localhost:8000/Index-1.htm'] #Semilla
    def __init__(self,limit,seed):
        # super(*args,**kwargs)
        global start_urls
        start_urls = [seed]
        self.limit =  int(limit)
        self.count_dict = {"compra-venta":0 ,"computadoras":0, "autos":0,"vivienda":0,"empleos":0}
        self.l_extractor = LinkExtractor(canonicalize=True)
        self.count = 0
    def parse(self, response):
        if self.count < self.limit:
            n = response.url.split("/")[-1]
            category = response.url.split("/")[-3]

            if n != "index.htm" and (category == "compra-venta" or category == "computadoras" or category == "autos" or category == "vivienda" or category == "empleos") :
                category_count = self.count_dict[category]
                # links = []
                if  category_count < int(self.limit/5):
                    self.count_dict[category] = category_count + 1
                    filename = category +"_"+ str(category_count)+'.html'
                    with open(f'crawler/websites/{filename}','wb') as f:
                        head = response.css("h1.headingText::text").extract()
                        text = head + response.css("span.showAdText::text").extract()
                        if len(text) > 0:
                            text = '\n'.join(text)
                        else:
                            text = ""
                    
                        pickle.dump((category,text),f)
                        self.count += 1 #Contador de la cantidad de paginas crawleadas
            links = self.l_extractor.extract_links(response)
            
            for link in links:
                yield response.follow(link.url, callback=self.parse, errback = self.errback)
        else:
            raise scrapy.exceptions.CloseSpider(reason = 'limit reached')

    def errback(self,response): #En caso de fallo decremento el contador ya que la pagina no se descargo
        self.count -= 1