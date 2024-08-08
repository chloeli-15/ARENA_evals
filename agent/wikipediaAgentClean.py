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
'''
First some useful functions
'''

def get_page(title):
    #Gets the wikipedia API page object for the page titled "title".
    return wikipedia.page(title, auto_suggest = False, redirect = True)

def tool_call_message(tool_call,content):
        return {
            "tool_call_id" : tool_call.id,
            "role" : "tool",
            "name" : tool_call.function.name,
            "content" : content
        }
def user_message(content):
    return {
        "role" : "user",
        "content" : content
    }



class WikipediaGame:
    def __init__(self, starting_page : str, goal_page : str, rules : list | type[None] = None):
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

    def get_page_summary(self, page):
        summary = page.summary[0:1000]
        return summary[0: summary.rindex(".")+1]


    def move_page(self, arguments):
        new_page = arguments["new_page"]
        if self.is_permitted_link(new_page):
            self.current_page = get_page(new_page)
            self.page_title_history.append(new_page)
            return True
        else:
            return False

    def get_history(self):
        return self.page_title_history


    def get_plain_content(self):
        return self.current_page.content

    def get_content(self,arguments):
        #Gives content with links
        content = self.current_page.content
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = content.replace(" " + word + " ", " " + f"<link>{word}</link>" + " ")
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = content.replace(" " + word + ",", " " + f"<link>{word}</link>" + ",")
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = content.replace(" " + word + ".", " " + f"<link>{word}</link>" + ".")
        return content

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
    def check_win(self):
        if self.current_page==self.goal_page:
            return True
        else:
            return False

a = WikipediaGame("Barack Obama", "India")
#%%

'''
Ensure that the wikipedia tool names correspond to callable functions on the WikipediaGame class (as this is how I'm going to handle them to avoid annoying things).
'''

WikipediaGameTools = [
            {
                "type" : "function",
                "function" : {
                    "name" : "get_content",
                    "description" : "Get all the content for the wikipedia page you are currently on.",
                    "parameters" : {
                        "type" : "object",
                        "properties" : {},
                        "required" : []
                    }
                }
            },
            {
                "type" : "function",
                "function" : {
                    "name" : "move_page",
                    "description" : "Changes your current page to a specified new page which is accessible via a link from the current page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "new_page": {
                                "type": "string",
                                "description": "The title of the new page you want to move to.",
                            },
                        },
                        "required": ["new_page"]
                    }
                }
            }
        ]

#%%

class WikipediaRacingAgent:
    def __init__(self, wikigame, model = "gpt-4o", tools = WikipediaGameTools):
        self.game=wikigame
        self.model = model
        self.default_system_message = {
                "role" : "system", 
                "content" : "You are a wikipedia-racing AI. Your goal is to reach " + self.game.get_goal() + ". Your current page is " + self.game.get_title() + ". The path you have taken so far is " + " -> ".join(self.game.get_history()) + "."
            }
        try:
            self.additional_rules_list = list(self.game.rules.keys())
        except:
            self.additional_rules_list = []
        self.additional_rules_string= self.generate_additional_rules_string()
        self.default_user_message={
                "role" : "user",
                "content" : "You are currently on page: " + self.game.current_page.title + ". Before using any functions, you should reason about which articles on your current page will make it easier to get to the article on " + self.game.goal_page.title + ". Make sure to reason through what your next step should be, but only your next step, don't plan your whole journey in advance, you never know what links might come up. In case you're unsure " + self.game.goal_page.title + " has the following summary: " + self.game.get_page_summary(self.game.goal_page) + "\nMake sure to read the content of the wikipedia page you are on, in order to choose the best possible link. Once you have chosen a link, you will not be able to go back to the current page, so choose the best option you can." + self.additional_rules_string
            }
        self.all_messages = [self.default_system_message, self.default_user_message]
        self.tools = tools
        self.response=None
        self.messages = [self.default_system_message, self.default_user_message]
    
    def generate_additional_rules_string(self):
        rules_list = self.additional_rules_list
        if len(rules_list) == 0:
            return ""
        elif len(rules_list) == 1:
            return ("There is an additional rule: " + rules_list[0])
        else:
            additional_rules_string = "There are some additional rules"
            for i in rules_list:
                additional_rules_string = additional_rules_string + " " + i + "."
            return additional_rules_string
    
    @retry_with_exponential_backoff
    def generate_response(self):
        new_response = client.chat.completions.create(
            model = self.model,
            messages = self.messages,
            tools = self.tools,
            tool_choice = "auto"
        )

        self.response = new_response.choices[0].message
        self.messages.append(self.response)
        self.all_messages.append(self.response)
        return self.response

    def get_response(self):
        return self.response

    def do_tool_calls(self, output):
        tool_calls = output.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                func = getattr(self.game, tool_call.function.name)
                arguments = json.loads(tool_call.function.arguments)
                new_response = func(arguments)
                self.messages.append(tool_call_message(tool_call,new_response))
                self.all_messages.append(tool_call_message(tool_call,new_response))
                if tool_call.function.name == "move_page" and self.game.is_permitted_link(arguments['new_page']):
                    return self.new_state()
                elif tool_call.function.name=="move_page" and not self.game.is_permitted_link(arguments['new_page']):
                    self.messages.append(user_message("That is not a valid link."))
                    self.all_messages.append(user_messages("That is not a valid link."))
                self.messages.append(user_message("What's your next step?"))
                self.all_messages.append(user_message("What's your next step?"))
                
                    

    def new_state(self):
        self.messages=[self.default_system_message,self.default_user_message]
        self.all_messages.extend([self.default_system_message,self.default_user_message])
        return "new state"







a = WikipediaGame("Barack Obama", "India")
#%%


# %%

game = WikipediaGame("Barack Obama", "India")
agent = WikipediaRacingAgent(game)
print(agent.messages[0]["content"])
print(agent.messages[1]["content"])
for i in range(0,3):
    response = agent.generate_response()
    print(response.content)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(tool_call.function.name)
    if agent.do_tool_calls(response) == "new state":
        new_state_string = ("-" * 50) + "\n\nNEW STATE\n\n" + " -> ".join(game.page_title_history) + "\n" + ("-"*50)
        print(new_state_string)
        print(agent.messages[0]["content"])
        print(agent.messages[1]["content"])
    else:
        print("User: What's your next step?")
        

#%%
print(agent.messages)








# %%
