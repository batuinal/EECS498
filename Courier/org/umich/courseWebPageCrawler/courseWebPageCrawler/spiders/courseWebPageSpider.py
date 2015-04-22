import os
import re
import urlparse

import scrapy
from scrapy.contrib.spiders import CrawlSpider

from courseWebPageCrawler.items import CourseWebPage


class CourseWebPageSpider(CrawlSpider):
    name = 'courseWebPageCrawler'
    start_urls = ["http://ocw.mit.edu/courses/",
                "http://cs.stanford.edu/courses/schedules/2014-2015.autumn.php",
                "http://www.math.caltech.edu/courses.html",
                "http://ee2.caltech.edu/courses/index.html",
                "http://www.cms.caltech.edu/academics/course_desc",
                "http://www.cs.princeton.edu/courses/schedule",
                "http://oyc.yale.edu/courses",
                "http://cs.brown.edu/courses/",
                "https://www.coursera.org/courses",
                "https://www.edx.org/course",
                "https://www.udacity.com/courses#!/all"
                  ]
    allowed_domains = ['edu', 'coursera.org', 'edx.org', 'udacity.com']
    
    def parse(self, response):
        self.log('A response from %s just arrived!' % response.url)
        item = CourseWebPage()
        item['name'] = response.xpath("//title/text()").extract()
        content = item['description'] = response.body
        responseurl = item['url'] = response.url
        yield item
        
        fileName = re.sub(r'[/:]', r'_', responseurl)
        fileName = fileName + '.html'
        self.writeToFile(fileName, content)
        for url in response.xpath('//a/@href').extract():
            if not self.isAllowed(url):
                continue
            if url.startswith('#'):
                continue
            url = urlparse.urljoin(response.url, url.strip())
            print url
            yield scrapy.Request(url, callback=self.parse)
                

    def writeToFile(self, filename, content):
        working_dir = 'data'        
        content_as_string = str(content)
        with open(working_dir + '/' + filename, 'w') as fout:
            fout.write(content_as_string)
    
    def isAllowed(self, url):
        if re.match("^(file|ftp|mailto)", url):
            return False
        r = re.compile(r'[\?][a-zA-Z0-9]*=')
        if len(r.findall(url)) > 0:
            return False
        r = re.compile(r'[a-zA-Z0-9]*\&[a-zA-Z0-9]*=')
        if len(r.findall(url)) > 0:
            return False
        if re.match("-.*(/[^/]+)/[^/]+\1/[^/]+\1/", url):
            return False
        if re.match(".*\\.(bmp|GIF|gif|jpe?g|png|tiff?|pdf|ico|ICO|xaml|pict|rif|pptx?|ps" +
            "|css|CSS|sit|SIT|eps|EPS|wmf|ppt|xls|XLS|GZ|rpm|tgz|exe|bmp|js" +
            "|mid|mp2|mp3|mp4|wav|wma|au|aiff|flac|ogg|3gp|aac|amr|au|vox" +
            "|avi|mov|mpe?g|ra?m|m4v|smil|wm?v|swf|aaf|asf|flv|mkv" +
            "|zip|rar|gz|7z|aac|ace|alz|apk|arc|arj|dmg|jar|lzip|lha)" +
            "(\\?.*)?$", url):
            return False
        return True