# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 07:20:26 2021

@author: Dong Hyun (Veo) Chae
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

#google search links that contain the youtube videos to be utilized 
links = ['https://www.google.com/search?q=youtube%3A+vaccine+hesitancy+news+channel&biw=1536&bih=722&tbm=vid&ei=a7wGYaPTM8n2hwORkJCgBA&oq=youtube%3A+vaccine+hesitancy+news+channel&gs_l=psy-ab-video.3...1407.3162.0.3278.13.10.1.0.0.0.223.1074.0j4j2.6.0....0...1c.1.64.psy-ab-video..9.0.0....0.3gCzVGw-Qf0',
         'https://www.google.com/search?q=youtube:+vaccine+hesitancy+news+channel&tbm=vid&ei=nroHYdmnEqWSr7wPiZ-c-Aw&start=10&sa=N&ved=2ahUKEwjZ7sf6gpLyAhUlyYsBHYkPB88Q8tMDegQIARBM&biw=1536&bih=722&dpr=1.25',
         'https://www.google.com/search?q=youtube:+vaccine+hesitancy+news+channel&tbm=vid&ei=67sHYbiNENPrwQOfuJyIDg&start=20&sa=N&ved=2ahUKEwj4rqqZhJLyAhXTdXAKHR8cB-E4ChDy0wN6BAgBEE4&biw=1536&bih=722&dpr=1.25',
         'https://www.google.com/search?q=youtube:+vaccine+hesitancy+news+channel&tbm=vid&ei=_bsHYbL3LcKvoATu7quwDg&start=30&sa=N&ved=2ahUKEwjy6ZKihJLyAhXCF4gKHW73CuY4FBDy0wN6BAgBEFA&biw=1536&bih=722&dpr=1.25',
         'https://www.google.com/search?q=youtube:+anti-vaccine+news+channel+coverage&rlz=1C1GCEU_enUS923US923&source=lnms&tbm=vid&sa=X&ved=2ahUKEwj9xYjOhJLyAhXSad4KHTVeDQoQ_AUoAXoECAEQAw&biw=1536&bih=722',
         'https://www.google.com/search?q=youtube:+anti-vaccine+news+channel+coverage&rlz=1C1GCEU_enUS923US923&tbm=vid&ei=ZrwHYcnyJ8HK-QbxwJr4Dg&start=10&sa=N&ved=2ahUKEwjJvZXUhJLyAhVBZd4KHXGgBu8Q8tMDegQIARBM&biw=1536&bih=722&dpr=1.25',
         'https://www.google.com/search?q=youtube:+anti-vaccine+news+channel+coverage&rlz=1C1GCEU_enUS923US923&tbm=vid&ei=ZrwHYcnyJ8HK-QbxwJr4Dg&start=20&sa=N&ved=2ahUKEwjJvZXUhJLyAhVBZd4KHXGgBu8Q8tMDegQIARBO&biw=1536&bih=722&dpr=1.25']

#function to request html source 
def get_page (link):
    page = requests.get(link)
    return page

    #function utilized
    page_0 = get_page(links[0])
    page_1 = get_page(links[1])
    page_2 = get_page(links[2])
    page_3 = get_page(links[3])
    page_4 = get_page(links[4])
    page_5 = get_page(links[5])
    page_6 = get_page(links[6])

#function to parse the html code of the website
def get_soup (page):
    soup = BeautifulSoup(page.text, 'html.parser')
    return soup

    #function utilized
    soup_0 = get_soup(page_0)
    soup_1 = get_soup(page_1)
    soup_2 = get_soup(page_2)
    soup_3 = get_soup(page_3)
    soup_4 = get_soup(page_4)
    soup_5 = get_soup(page_5)
    soup_6 = get_soup(page_6)

#function that allows for the youtube links to be extracted from the soup
def url_extractor(page,soup):
        #first looked for all "<a>"
    tags = soup.find_all('a')
        #then looked for "<a>" that contains href
    tags = [tag for tag in tags if tag.has_attr('href')]
    links = [tag['href'] for tag in tags]
        #saving all the website links that could be found
    links = pd.DataFrame(links)
    links.columns = ["link"]
        #only sorting the website with youtube.com in it
    links = links[links["link"].str.contains("youtube.com")]
        #first 7 characters is a filler for coding purposes so deleted first 7 chracters from all rows
    links['link'] = links['link'].str[7:]
        #the %3Fv%3D stands for ?v= when it comes to website coding --> looked it up on google
    links['link'] = links['link'].str.replace('%3Fv%3D', '?v=')
        #deleted everything that comes after (including) & --> only leaving the URL of the Youtube videos
    links['link'] = links['link'].apply(lambda x: x.split('&')[0])
        #checking to see if there are duplicates and deleting if there are
    links = links.drop_duplicates(keep= 'first', inplace=False)
        #re-indexing
    links.index = np.arange(1, len(links) + 1)
    return links

    #function utilized
    links_0 = url_extractor(page_0, soup_0)
    links_1 = url_extractor(page_1, soup_1)
    links_2 = url_extractor(page_2, soup_2)
    links_3 = url_extractor(page_3, soup_3)
    links_4 = url_extractor(page_4, soup_4)
    links_5 = url_extractor(page_5, soup_5)
    links_6 = url_extractor(page_6, soup_6)
    
    #appending all the links gathered into links_total
    links_total = links_0.append([links_1,links_2,links_3,links_4,links_5,links_6])
    #making sure there are no duplicate videos
    links_total = links_total.drop_duplicates(keep= 'first', inplace=False) 
    #resetting the index
        #used an index starting from 1 because it will make it easier when storing the scripts into txt files
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
            print('manual caption')a
        else:
            print('auto-caption')
    else:
        print('no english captions')


#reading in the empty txt files that were created to store the scripts
    #since there were 55 videos that were found, 55 txt files were created
    script = pd.DataFrame(["1.txt","2.txt","3.txt","4.txt","5.txt","6.txt","7.txt","8.txt","9.txt","10.txt","11.txt","12.txt","13.txt","14.txt","15.txt","16.txt","17.txt","18.txt","19.txt","20.txt",
                       "21.txt","22.txt","23.txt","24.txt","25.txt","26.txt","27.txt","28.txt","29.txt","30.txt",
                       "31.txt","32.txt","33.txt","34.txt","35.txt","36.txt","37.txt","38.txt","39.txt","40.txt",
                       "41.txt","42.txt","43.txt","44.txt","45.txt","46.txt","47.txt","48.txt","49.txt","50.txt",
                       "51.txt","52.txt","53.txt","54.txt","55.txt", "56.txt", "57.txt"])
    script.columns = ["script"]
    #also used index starting from 1 to make things eaiser when storing scripts
    script.index = np.arange(1, len(script) + 1)

#changed the working directory to ensure that calling for the txt files do not have a problem
os.chdir("C:/Users/dchae2/Desktop/youtube scripts")

# x mark on the right side of the codes represent that the video does not support subtitles
    #extraction complete
    captions_test02(links_total['link'][1], script['script'][1])
    captions_test02(links_total['link'][4], script['script'][4])
    captions_test02(links_total['link'][5], script['script'][5])
    captions_test02(links_total['link'][6], script['script'][6])
    captions_test02(links_total['link'][8], script['script'][8])
    captions_test02(links_total['link'][9], script['script'][9])
    captions_test02(links_total['link'][10], script['script'][10])
    captions_test02(links_total['link'][11], script['script'][11])
    captions_test02(links_total['link'][12], script['script'][12])
    captions_test02(links_total['link'][13], script['script'][13])
    captions_test02(links_total['link'][14], script['script'][14])
    captions_test02(links_total['link'][15], script['script'][15])
    captions_test02(links_total['link'][16], script['script'][16])
    captions_test02(links_total['link'][17], script['script'][17])
    captions_test02(links_total['link'][22], script['script'][22])
    captions_test02(links_total['link'][23], script['script'][23])
    captions_test02(links_total['link'][25], script['script'][25])
    captions_test02(links_total['link'][26], script['script'][26])
    captions_test02(links_total['link'][27], script['script'][27])
    captions_test02(links_total['link'][28], script['script'][28])
    captions_test02(links_total['link'][30], script['script'][30])
    captions_test02(links_total['link'][35], script['script'][35])
    captions_test02(links_total['link'][37], script['script'][37])
    captions_test02(links_total['link'][38], script['script'][38])
    captions_test02(links_total['link'][39], script['script'][39])
    captions_test02(links_total['link'][40], script['script'][40])
    captions_test02(links_total['link'][41], script['script'][41])
    captions_test02(links_total['link'][42], script['script'][42])
    captions_test02(links_total['link'][43], script['script'][43])
    captions_test02(links_total['link'][44], script['script'][44])
    captions_test02(links_total['link'][45], script['script'][45])
    captions_test02(links_total['link'][46], script['script'][46])
    captions_test02(links_total['link'][47], script['script'][47])
    captions_test02(links_total['link'][48], script['script'][48])
    captions_test02(links_total['link'][51], script['script'][51])
    captions_test02(links_total['link'][52], script['script'][52])
    captions_test02(links_total['link'][53], script['script'][53])
    captions_test02(links_total['link'][54], script['script'][54])
    captions_test02(links_total['link'][57], script['script'][57])
    captions_test02(links_total['link'][55], script['script'][55]) 
    captions_test02(links_total['link'][3], script['script'][3]) 
    captions_test02(links_total['link'][18], script['script'][18]) 
    captions_test02(links_total['link'][24], script['script'][24]) 
    captions_test02(links_total['link'][31], script['script'][31]) 
    captions_test02(links_total['link'][33], script['script'][33]) 
    captions_test02(links_total['link'][49], script['script'][49]) 
    captions_test02(links_total['link'][34], script['script'][34]) 
    
    #extraction unavailable
    captions_test02(links_total['link'][7], script['script'][7]) #no english caption
    captions_test02(links_total['link'][19], script['script'][19]) #no english caption
    captions_test02(links_total['link'][20], script['script'][20]) #no english caption
    captions_test02(links_total['link'][29], script['script'][29]) #no english caption
    captions_test02(links_total['link'][36], script['script'][36]) #no english caption
    captions_test02(links_total['link'][2], script['script'][2]) #
    captions_test02(links_total['link'][21], script['script'][21]) #
    captions_test02(links_total['link'][32], script['script'][32]) #
    captions_test02(links_total['link'][50], script['script'][50]) #
    captions_test02(links_total['link'][56], script['script'][56]) #

#deleting all the script non-downloadable links
links_total = links_total.drop([7,19,20,29,36,2,21,32,50,56], axis = 0)
links_total.index = np.arange(0, len(links_total))
script = script.drop([7,19,20,29,36,2,21,32,50,56], axis = 0)
script.index = np.arange(0, len(script))

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

final = pd.DataFrame()
for i in range(len(script['script'])):
    a = script_clean(script['script'][i])
    final = final.append(a)    

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

#exporting the file to csv
finalized.to_csv('C:/Users/dchae2/Desktop/Youtube_finalized.csv')
 
#trying to scrape the media bias chart but not successful --> contacted the website for permission to
html = pd.read_html("https://www.allsides.com/media-bias/media-bias-ratings#ratings")
movie_ratings=html[0]
movie_ratings.info()