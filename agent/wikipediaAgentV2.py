#%%
import json
import os
os.chdir("c:\\Users\\styme\\OneDrive\\Documents\\AI STUFF\\Model Written Evals\\Code Replication\\ARENA_evals\\agent")
import wikipedia
import re
from openai import OpenAI
from utils import establish_client_OpenAI
from utils import retry_with_exponential_backoff
from pprint import pprint
import numpy as np
from country_list import countrylist
countrylist = [i[1] for i in countrylist]

client = establish_client_OpenAI()


'''
TODO: 
    - Build a more robust ReAct framework into the agent calls. !
        - Generate + Evaluate strategies. (Probably as part of ReAct Framework) Done!
        - Get agent to use tools more effectively. Done-ish!

    - Build trialling paths. (either tool or special call after the first "thought") (e.g. generate 3 "guess" paths, and I will tell you where/when they go wrong.) Done!

    TODO: Ban list pages, table pages, etc since they aren't allowed so often lead to stubs/purely circular routes :(

    - Reasoning histories (hard to know how to implement this). Maybe build some sort of Meta-Agent to handle this.

    - New Tools:
        - Build get_plain_content_of_any_wikipedia_page tool
        - Build get_page_summary_before_moving tool
        - Build get_arbitrary_page_summary
        - Build








'''



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
    
    def get_links_of_page(page_title):
        page = wikipedia.page(page_title, redirect= True, auto_suggest=False)
        all_links = page.links
        content = page.content
        permitted_links = []
        for i in all_links:
            if i in content:
                permitted_links.append(i)
        return permitted_links
    
    def test_path(self, arguments : dict) -> str:
        path = arguments["path"]
        if path[0].strip() != self.current_page.title:
            return "ERROR: The title of the start page of this path is not the title of your current page."
        if path[-1].strip()!= self.goal_page.title:
            return "ERROR: The title of the final page of this path is not the title of your goal page."
        for i in range(len(path)-1):
            if path[i+1].strip() in get_links_of_page(path[i]):
                continue
            else:
                return "This path works until " + path[i+1].strip() + ", which is not accessible from " +path[i].strip()
        return "This path completely works."
    
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
            },
            {
                "type" : "function",
                "function" : {
                    "name" : "test_path",
                    "description" : "Accepts a test path string in the form \"current_page -> new_page1 -> new_page2 -> ... -> goal_page\" and returns where the path goes wrong if the path does not work, or \"success\" if the path works.",
                    "parameters" : {
                        "type" : "object",
                        "properties": {
                            "path" : {
                                "type" : "string",
                                "description" : "The path you want to test."
                            },
                        },
                        "required" : ["path"]
                    }
                }
            }
        ]

#%%
'''
import re
import wikipedia

ex = "<path1> Barack Obama -> Indonesia -> India</path1> and my second path is <path2> Barack Obama -> United Nations -> India </path2> and my third path is <path3>2001 Strathkelvin and Bearsden by-election -> Scottish Parliament -> Scotland -> United Kingdom -> Summer Olympic Games -> 1980 Summer Olympics -> Boxing -> Olympic Games -> 1980 Summer Olympics -> Boxing -> Olympic boxing </path3>"
def get_paths(output : str) -> list[str]:
    pathdict=[] # a silly name as a sort of joke
    for path in re.findall(r'<path[\d]*>.*?</path', output):
        pathlist_dict.append(path[7:-6].split("->"))
    return pathlist_list

path_list = get_paths(ex)
print(path_list[0])

def get_links_of_page(page_title):
        page = wikipedia.page(page_title, redirect= True, auto_suggest=False)
        all_links = page.links
        content = page.content
        permitted_links = []
        for i in all_links:
            if i in content:
                permitted_links.append(i)
        return permitted_links

def evaluate_path(path : list) -> str:
    for i in range(len(path)-1):
        if path[i+1].strip() in get_links_of_page(path[i]):
            continue
        else:
            return "This path works until " + path[i+1].strip() + ", which is not accessible from " +path[i].strip()
    return "This path completely works."
print(evaluate_path(path_list[2]))

'''
tool_string =  "You have access to three functions:\n\nget_content: This outputs the wikipedia content of the current page you are in, with any available links wrapped in <link> </link> tags. \nmove_page: This takes a page title as argument, and lets you move page to any available link from your current page.\ntest_path: this takes a trial path in the form current_page -> page_1 -> page_2 -> ... -> goal_page as argument, and outputs success if the path works, or tells you where the path goes wrong if the path doesn't work.\n"
#%%
class WikipediaRacingAgent:
    def __init__(self, wikigame, model = "gpt-4o", tools = WikipediaGameTools):
        self.game=wikigame
        self.state_count=0
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
                "content" : "[GAME SUMMARY]\n\nYou are currently on page: " + self.game.current_page.title + ". The path you have taken so far is " + " -> ".join(self.game.get_history()) + ". When you use get_content, any pages you can access will be wrapped in <link> tags. You should try to use the test_path function to get information about your plans." +"\n[GAME SUMMARY]\nWhen coming up with a strategy, pay attention to the previous path you've taken, and if it doesn't seem to be working out then try something else. In case you're unsure, " + self.game.goal_page.title + " has the following summary: \n\n[Begin Target Page Summary]\n" + self.game.get_page_summary(self.game.goal_page) + "\n[End Target Page Summary]" + self.additional_rules_string
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
    def generate_response(self, tools=True):
        if tools:
            new_response = client.chat.completions.create(
                model = self.model,
                messages = self.messages,
                tools = self.tools,
                tool_choice = "auto"
            )
        else:
            new_response = client.chat.completions.create(
                model = self.model,
                messages = self.messages,
            )
        self.response = new_response.choices[0].message
        #self.messages.append(self.response)
        #self.all_messages.append(self.response)
        return self.response
    
    def ReAct_framework_generate_response(self, self_evaluate=True):
        num_thoughts = 1
        num_actions = 1
        response_list=[]
        for i in range(num_thoughts+num_actions):
            if i%2 == 0:
                self.messages.append(user_message(
                    "Think about what plans you want to take. Generate at least two (possibly more if you can) candidate plans. Then decide which plan you think is best."
                ))
                new_response = self.generate_response(tools=False)
                response_list.append(new_response)
                self.response = new_response
                self.messages.append(self.response)
                self.all_messages.append(self.response)
            else:
                self.messages.append(user_message(
                    "Now act on your best plan."
                ))
                new_response = self.generate_response(tools=True)
                response_list.append(new_response)
                self.response = new_response
                self.messages.append(self.response)
                self.all_messages.append(self.response)
                return response_list
        
    def get_response(self):
        return self.response

    def do_tool_calls(self, output):
        tool_calls = output.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                func = getattr(self.game, tool_call.function.name)
                if tool_call.function.name == "get_content":
                    arguments=None
                else:
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
                
                    

    def new_state(self):
        if self.state_count>=4:
            high_state_string="\nYou should pay attention to the path you have taken so far. If you have been going around certain pages in circles, and you keep returning to the same page, then try a plan that will take you in a different direction."
        else:
            high_state_string=""
        if self.game.current_page.title in game.get_history()[:-1]:
            already_visited_string = "\n\nNotice that you have *already* visited this page. You can see what went wrong last time you visited this page in the \"paths you've taken so far.\" Make sure you react to this appropriately, and don't just try the same thing you already tried before."
        else:
            already_visited_string=""
        self.default_system_message = {
                "role" : "system", 
                "content" : "You are a wikipedia-racing AI. Your goal is to reach " + self.game.get_goal() + ". Your current page is " + self.game.get_title() + "."
            }
        self.default_user_message={
                "role" : "user",
                "content" : "[GAME SUMMARY]\n\nYou are currently on page: " + self.game.current_page.title + ". The path you have taken so far is " + " -> ".join(self.game.get_history()) + ". When you use get_content, any pages you can access will be wrapped in <link> tags. You should try to use the test_path function to get information about your plans." +"\n[GAME SUMMARY]\n When coming up with a strategy, pay attention to the previous path you've taken, and if it doesn't seem to be working out then try something else. In case you're unsure, " + self.game.goal_page.title + " has the following summary: \n\n[Begin Target Page Summary]\n" + self.game.get_page_summary(self.game.goal_page) + "\n[End Target Page Summary]" + high_state_string + self.additional_rules_string + already_visited_string
                }
        self.messages=[self.default_system_message,self.default_user_message]
        self.all_messages.extend([self.default_system_message,self.default_user_message])
        self.state_count+=1

#%%

# %%
'''
Games of varying difficulty:

Obama -> India

Colin McEvedy -> India

Obama -> Tsetse fly (the MVP agent can solve this in 18 steps (including start and end), this agent can do it in 12)

Doyle Brunson -> Montu


'''



game = WikipediaGame("Doyle Brunson", "Montu")
agent = WikipediaRacingAgent(game, model = "gpt-4o")
#%%
def agent_loop():
    if game.check_win():
        print("Success")
        return ""

    responses = agent.ReAct_framework_generate_response()
    print(responses[0].content)
    print(responses[1])
    if responses[1].tool_calls:
        for tool_call in responses[1].tool_calls:
            if tool_call.function.name == "move_page":
                print(tool_call.function.name + " " + tool_call.function.arguments)
            elif tool_call.function.name == "get_content":
                print(tool_call.function.name + " " + game.current_page.title)
            elif tool_call.function.name == "test_path":
                print(tool_call.function.name + " " + json.loads(tool_call.function.arguments)["path"])
    if agent.do_tool_calls(responses[1]) == "new state":
        new_state_string = ("-" * 50) + "\n\nNEW STATE\n\n" + " -> ".join(game.page_title_history) + "\n\n" + ("-"*50)
        print(new_state_string)
        print(agent.messages[0]["content"])
        print(agent.messages[1]["content"])


#%%
for i in range(0,20):
    agent_loop()


#%%
print(" -> ".join(game.get_history()[:-1]))


#%%
for i in agent.all_messages:
    try: print(i['content']) 
    except: print(i.content)
#%%
print(game.is_permitted_link("Tamil Nadu"))
print(game.get_links())
print(game.current_page.content)





# %%
