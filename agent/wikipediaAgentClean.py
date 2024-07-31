#%%
import json
import os
os.chdir("c:\\Users\\styme\\OneDrive\\Documents\\AI STUFF\\Model Written Evals\\Code Replication\\ARENA_evals\\agent")
import wikipedia
from openai import OpenAI
from utils import establish_client_OpenAI
from utils import retry_with_exponential_backoff
from pprint import pprint
import numpy as np
from country_list import countrylist
countrylist = [i[1] for i in countrylist]

client = establish_client_OpenAI()
#%%

def get_page(title):
    #Gets the wikipedia API page object for the page titled "title".
    return wikipedia.page(title, auto_suggest = False, redirect = True)

class WikipediaGame:
    def __init__(self, starting_page : str, goal_page : str,rules : list[dict[str, str]] | type(None) = None):
        '''
        Initialises the wikipedia game object

        starting_page is the page the agent starts on.

        goal_page is the page the agent is trying to get to.

        rules is a list of dictionaries specifying any additional rules along with a description of that rule to be fed to the agent.

        Rules that are handled currently:
            - "no country" bans country articles.
            - "no backtrack" bans backtracking.
        '''
        self.page_title_history = [starting_page]
        self.starting_page = get_page(starting_page)
        self.goal_page = get_page(goal_page)
        self.rules = rules
        self.current_page = get_page(starting_page)

    def move_page(self, new_page):
        self.current_page = get_page(new_page)

    def get_content(self):
        return self.current_page.content

    def get_content_with_links(self):
        content = self.get_content()
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = content.replace(" " + word + " ", " " + f"<link>{word}</link>" + " ")
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = content.replace(" " + word + ",", " " + f"<link>{word}</link>" + ",")
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = content.replace(" " + word + ".", " " + f"<link>{word}</link>" + ".")

    def get_links(self):
        all_links = self.current_page.links
        content = self.current_page.content
        permitted_links = []
        for i in all_links:
            if i in content:
                permitted_links.append(i)
        return permitted_links

    def is_permitted_link(self, link):
        if link in self.get_links():
            return True
        else:
            return False

    def get_title(self):
        return self.current_page.title

    def get_start(self):
        return self.starting_page.title

    def get_goal(self):
        return self.goal_page.title


class Agent:
    def __init__(self):
        self.system_message
        self.user_message
        self.tools



a = WikipediaGame("Barack Obama", "India")



# %%
a.get_goal()
a.get_content()



# %%
