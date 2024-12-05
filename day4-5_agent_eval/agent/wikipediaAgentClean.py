#%%
import json
import os
os.chdir("c:\\Users\\styme\\OneDrive\\Documents\\AI STUFF\\Model Written Evals\\Code Replication\\ARENA_evals\\agent")
import wikipedia
from openai import OpenAI
from anthropic import Anthropic
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
        self.page_title_history = [starting_page] # History of pages that have been visited.
        self.starting_page = get_page(starting_page) # page the game starts on
        self.goal_page = get_page(goal_page) # page the game ends on
        self.rules = rules # any additional rules, (no countries, no articles above a certain length, no backtracking etc. Need to add functionality to deal with these which I haven't done yet)
        self.current_page = get_page(starting_page) # current page the game state is on.


    def get_page_summary(self, page):
        '''
        Gets summary of a wikipedia page, to the last full stop within the first 500 characters.
        '''
        summary = page.content[0:500]
        return summary[0: summary.rindex(".")+1]



    def move_page(self, arguments):
        '''
        The tool which will be used when we want to move from Page A to Page B
        '''
        new_page = arguments["new_page"]
        if self.is_permitted_link(new_page):
            self.current_page = get_page(new_page)
            self.page_title_history.append(new_page)
            return "Moving page to " + new_page
        else:
            return "Couldn't move page to " + new_page

    def get_history(self):
        '''
        Gets the histroy
        '''
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
                    "description" : "Get all the content for the wikipedia page you are currently on. Anything which corresponds to a link you can select will be wrapped in <link></link> tags.",
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
                    "description" : "Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.",
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
                "content" : "You are a wikipedia-racing AI. Your goal is to reach " + self.game.get_goal() + ". Your current page is " + self.game.get_title() + "."
            }
        try:
            self.additional_rules_list = list(self.game.rules.keys())
        except:
            self.additional_rules_list = []
        self.additional_rules_string= self.generate_additional_rules_string()
        self.default_user_message={
                "role" : "user",
                "content" : "You are currently on page: " + self.game.current_page.title + ". Make sure you start by reasoning about what steps you should take to get to the article on " + self.game.goal_page.title + ". When coming up with a strategy, make sure to pay attention to the path you have already taken, and if your current strategy doesn't seem to be working out, try something else. In case you're unsure, " + self.game.goal_page.title + " has the following summary: \n\n[Begin Summary]\n" + self.game.get_page_summary(self.game.goal_page) + "\n[End Summary]" + self.additional_rules_string
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
            additional_rules_string = "There are some additional rules:"
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

                if tool_call.function.name == "move_page" and "Moving page" in new_response:
                    self.new_state()
                    return "new state"

                elif tool_call.function.name=="move_page" and "Couldn't" in new_response:
                    self.messages.append(user_message("That is not a valid link."))
                    self.all_messages.append(user_message("That is not a valid link."))

            self.messages.append(user_message("What's your next step?"))
            self.all_messages.append(user_message("What's your next step?"))
                
                    

    def new_state(self):
        self.default_system_message = {
                "role" : "system", 
                "content" : "You are a wikipedia-racing AI. Your goal is to reach " + self.game.get_goal() + ". Your current page is " + self.game.get_title() + "."
            }
        self.default_user_message={
                "role" : "user",
                "content" : "You are currently on page: " + self.game.current_page.title + ". The path you have taken so far is " + " -> ".join(self.game.get_history()) + ". " + "Make sure you start by reasoning about what steps you should take to get to the article on " + self.game.goal_page.title + ". When coming up with a strategy, make sure to pay attention to the path you have already taken, and if your current strategy doesn't seem to be working out, try something else. In case you're unsure " + self.game.goal_page.title + " has the following summary: \n\n [Begin Summary] \n" + self.game.get_page_summary(self.game.goal_page) + "\n[End Summary]" + self.additional_rules_string
            }
        self.messages=[self.default_system_message,self.default_user_message]
        self.all_messages.extend([self.default_system_message,self.default_user_message])

#%%


# %%

game = WikipediaGame("For To Next", "Mexichromis trilineata")
agent = WikipediaRacingAgent(game, model = "claude-sonnet-3-5")
#%%
print(agent.messages[0]["content"])
print(agent.messages[1]["content"])
def agent_loop():
    if game.check_win():
        print("Success")
        return ""

    response = agent.generate_response()
    print(response.content)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(tool_call.function.name)
    if agent.do_tool_calls(response) == "new state":
        new_state_string = ("-" * 50) + "\n\nNEW STATE\n\n" + " -> ".join(game.page_title_history) + "\n\n" + ("-"*50)
        print(new_state_string)
        print(agent.messages[0]["content"])
        print(agent.messages[1]["content"])
    else:
        print("User: What's your next step?")


#%%
for i in range(0,10):
    agent_loop()





#%%
print(agent.messages)
print(agent.all_messages)
#%%
print(game.is_permitted_link("Tamil Nadu"))
print(game.get_links())







# %%
