#%%
import torch as t
import wikipedia
from openai import OpenAI
from utils import establish_client_OpenAI
from pprint import pprint
page_object = wikipedia.page("Iły, Masovian Voivodeship", auto_suggest=False,redirect=True)
pprint(page_object)
content = page_object.content
page_links = []
for i in page_object.links:
    if i in content:
        page_links.append(i)
    else:
        pass
print(page_links)
print(page_object.content)
# %%
'''
wikipedia.page returns a page_object type with most useful properties being:
    - content: A string of all the text on the wikipedia article.
    - links: Contains all the links on the article page. (Lots of these links wouldn't be allowed in classic wikipedia racing, this is why I'm searching for those links that are also named in the Content section)

'''

client = establish_client_OpenAI()

def get_current_page_content(current_page_title):
    #Outputs the content of the wikipedia page
    return wikipedia.page(current_page_title, auto_suggest=False, redirect=True).content

def get_all_page_links(current_page_title):
    return wikipedia.page(current_page_title, auto_suggest = False, redirect = True).links

def get_current_page_links(current_page_title):
    #Our
    allLinks = wikipedia.page(current_page_title, auto_suggest=False, redirect = True).links
    content = wikipedia.page(current_page_title, auto_suggest=False, redirect=True).content
    permittedLinks = []
    for i in allLinks:
        if i in content:
            permittedLinks.append(i)
    return permittedLinks


def check_if_permitted_link(link, current_page_title):
    allLinks = wikipedia.page(current_page_title, auto_suggest=False, redirect = True).links
    content = wikipedia.page(current_page_title, auto_suggest=False, redirect=True).content
    permittedLinks = []
    for i in allLinks:
        if i in content:
            permittedLinks.append(i)
    if link in permittedLinks:
        return True
    else:
        return False

print(len(get_current_page_links("India")))
print(len(get_all_page_links("India")))
print(get_current_page_content("India"))






# %%
#Runs one iteration of the game
messages = [{

        "role" : "system",
        "content" : "You are a Wikipedia-racing AI. You are given two articles and your goal is to use links on the wikipedia page to hop from article to article. You start at the START article, and aim to reach the END article."
    },
    {
        "role" : "assistant",
        "content" : "You are currently on page: **PAGE**. The path you have taken so far is **PATH**"
    },    
]
wikipedia_race_tools = [
    {
        "type" : "function",
        "function" : {
            "name" : "get_current_page_content",
            "description" : "Get the "
        }
    }
]

client.chat.completions.create(


)



#%%
