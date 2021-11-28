# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 07:54:40 2021

@author: dchae2
"""

#YOUTUBE VIDEO SCRIPT WEB SCRAPING

pip install youtube_dl
import os
import numpy as np
import youtube_dl
import requests
import re
import json
import pandas as pd
import requests                     
from bs4 import BeautifulSoup     
from pandas import DataFrame as df  

links_total = pd.DataFrame(['https://www.youtube.com/watch?v=yLo6IYQ4wsg', 'https://www.youtube.com/watch?v=Awz2uxw9Y8g', 'https://www.youtube.com/watch?v=9amSfpdwyj4', 'https://www.youtube.com/watch?v=h9WiTXYAQkg', 'https://www.youtube.com/watch?v=1nR6pqPulto', 'https://www.youtube.com/watch?v=A7vpcGVnPg0', 'https://www.youtube.com/watch?v=9ZQGr4vmKjo', 'https://www.youtube.com/watch?v=yqPhYG9tj38'])
links_total.columns = {'link'}  
links_total.index = np.arange(1, len(links_total) + 1)
    
#extracting captions from the links that was just extracted above    
def captions_test02(url, file_name):
    ydl = youtube_dl.YoutubeDL({'forceid': True, 'forcetitle': True,'writesubtitles': True, 'allsubtitles': True, 'writeautomaticsub': True})
    res = ydl.extract_info(url, download=False)
    if res['requested_subtitles'] and res['requested_subtitles']['en']:
        print('Subtitles are from ' + res['requested_subtitles']['en']['url'])
        yielded = requests.get(res['requested_subtitles']['en']['url'], stream=True)
        written_script = open(file_name, "w")
        new = re.sub(r'\d{2}\W\d{2}\W\d{2}\W\d{3}\s\W{3}\s\d{2}\W\d{2}\W\d{2}\W\d{3}','',yielded.text)
        written_script.write(new)
        written_script.close()
        if len(res['subtitles']) > 0:
            print('manual caption')
        else:
            print('auto-caption')
    else:
        print('no english captions')


#reading in the empty txt files that were created to store the scripts
    #since there were 55 videos that were found, 55 txt files were created
    script = pd.DataFrame(["1.txt","2.txt","3.txt","4.txt","5.txt","6.txt","7.txt","8.txt"])
    script.columns = ["script"]
    #also used index starting from 1 to make things eaiser when storing scripts
    script.index = np.arange(1, len(script) + 1)

#changed the working directory to ensure that calling for the txt files do not have a problem
os.chdir("C:/Users/dchae2/Desktop/youtube scripts")

# x mark on the right side of the codes represent that the video does not support subtitles
    #extraction complete
    captions_test02(links_total['link'][1], script['script'][1])
    captions_test02(links_total['link'][2], script['script'][2])
    captions_test02(links_total['link'][3], script['script'][3])
    captions_test02(links_total['link'][4], script['script'][4])
    captions_test02(links_total['link'][5], script['script'][5])
    captions_test02(links_total['link'][6], script['script'][6])
    captions_test02(links_total['link'][7], script['script'][7])
    captions_test02(links_total['link'][8], script['script'][8])


#cleaning the script
    #getting rid of all the unnecessary sound effects, time capture, spacing, etc.
    #some captions were repeated twice when in extraction, thus deleted the duplicates
    #then transfromed the script into a list so that it can be appended into a complete file
def script_clean(file):
    new_name = open(file, "r")
    new_name = new_name.readlines()
    new_name = pd.DataFrame(new_name)
    new_name.columns = ["script"]
    new_name = new_name[~new_name.script.str.contains("%")]
    new_name = new_name[~new_name.script.str.contains("<")]
    new_name = new_name[~new_name.script.str.contains("&")]
    new_name = new_name[new_name.script != "\n"]
    new_name = new_name[new_name.script != " \n"]
    new_name = new_name[new_name.script != "  \n"]
    new_name = new_name[new_name.script != "   \n"]
    new_name = new_name.drop_duplicates(keep= 'first', inplace=False)
    new_name = pd.DataFrame({'script': new_name['script'].sum(axis = 0)}, index = [0])
    return new_name
    
#reading in the clean scripts 
script['script'] = 'C:/Users/dchae2/Desktop/youtube scripts/'+script['script']

script.index = np.arange(0, len(script))
links_total.index = np.arange(0, len(links_total))


final = pd.DataFrame()
for i in range(len(script['script'])):
    a = script_clean(script['script'][i])
    final = final.append(a)  

final.index = np.arange(0, len(final))

#fuction for collecting the names,likes,dislikes, and title of the video providers for each caption/video
def info_extractor(url):
    uploader_name = {}
    with youtube_dl.YoutubeDL(uploader_name) as ydl:
        data = ydl.extract_info(url, download=False) 
        important = pd.DataFrame({'uploader': data['uploader'], 'like_count': data['like_count'], 'dislike_count': data['dislike_count'], 'title': data['title']},index = [0]) 
        return important                        

#collecting all the data using info_extractor on all 43 videos utilized
info_total = pd.DataFrame()
for i in range(len(links_total)):
    a = info_extractor(links_total['link'][i])
    info_total = info_total.append(a)

#resetting the index for join with scripts
info_total.index = np.arange(0, len(info_total))

#Final dataset with info on: Uploader, Like, Dislike, Title, Script
finalized = info_total.join(final)
finalized['scripts'] = finalized['script']
finalized = finalized.drop('script', 1)

#exporting the file to csv
finalized.to_csv('C:/Users/dchae2/Desktop/Right_Youtube_finalized.csv', index = False)
 
#trying to scrape the media bias chart but not successful --> contacted the website for permission to
html = pd.read_html("https://www.allsides.com/media-bias/media-bias-ratings#ratings")
movie_ratings=html[0]
movie_ratings.info()