#%%
import json
import os
# os.chdir("c:\\Users\\styme\\OneDrive\\Documents\\AI STUFF\\Model Written Evals\\Code Replication\\ARENA_evals\\day4-5_agent_eval\\agent")
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
#%%


'''
First some useful functions
'''

def get_page(title):
    #Gets the wikipedia API page object for the page titled "title".
    return wikipedia.page(title, auto_suggest = False, redirect = True)

def tool_call_message(tool_call,content):
        #Easily creates tool call messages
        return {
            "tool_call_id" : tool_call.id,
            "role" : "tool",
            "name" : tool_call.function.name,
            "content" : content
        }
def user_message(content):
    #Easily creates user messages
    return {
        "role" : "user",
        "content" : content
    }

class WikipediaGame:
    def __init__(self, starting_page : str, goal_page : str, rules : list | type[None] = None):
        '''
        Initialises the wikipedia game object

        starting_page is the page the agent starts on.

        current_page is the page the agent is currently on.

        goal_page is the page the agent is trying to get to.

        rules is a list of dictionaries specifying any additional rules along with a description of that rule to be fed to the agent
            - rules aren't handled right now.
        '''
        self.page_title_history = [starting_page] # History of pages that have been visited.
        self.starting_page = get_page(starting_page) # page the game starts on
        self.goal_page = get_page(goal_page) # page the game ends on
        self.rules = rules # any additional rules, (no countries, no articles above a certain length, no backtracking etc. Need to add functionality to deal with these which I haven't done yet)
        self.current_page = get_page(starting_page) # current page the game state is on.


    def get_page_summary(self, page):
        '''
        Gets summary of a wikipedia page, to the last full stop within the first 700 characters (so I don't cut off the middle of sentences).
        '''
        summary = page.content[0:700]
        return summary[0: summary.rindex(".")+1]



    def move_page(self, arguments):
        '''
        The tool which the agent will use when it wants to move from Page A to Page B
        '''
        new_page = arguments["new_page"] # Have to do .lower() in lots of places, or case issues arise.

        #Check that the page is permitted, if so move there. Otherwise say we can't move there.
        if self.is_permitted_link(new_page):
            self.current_page = get_page(new_page)
            self.page_title_history.append(new_page)
            return "Moving page to " + new_page
        else:
            return "Couldn't move page to " + new_page

    def get_history(self):
        '''
        Returns the list of the titles of previous pages that have been visited. (NOT ChatHistory !)
        '''
        return self.page_title_history

    def get_content(self,arguments):
        '''
        Gives content of wikipedia page with <link> tags. There's probably a better way to do this. Didn't use .replace() because of case sensitivity annoyances, so had to use regular expressions (re) library with the flag re.I
        '''
        content = self.current_page.content
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = re.sub(" " + word + " ", " " + f"<link>{word}</link>" + " ",content, flags = re.I)
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = re.sub(" " + word + ",", " " + f"<link>{word}</link>" + ",", content, flags=re.I)
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = re.sub(" " + word + ".", " " + f"<link>{word}</link>" + ".", content, flags=re.I)
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = re.sub("(" + word + ")", "(" + f"(<link>{word}</link>)" + ")", content, flags = re.I)
        for word in sorted(self.get_links(), key=len, reverse = True):
            content = re.sub(" " + word + "s", " " + f"(<link>{word}</link>)" + "s", content, flags = re.I)
        return content
    
    def get_links_of_page(self, page_title):
        '''
        Gets a list of links of a given page. This won't be called by the agent, since it can't get arbitrary links, but is necessary for the reflexion test_path function, as we need to check that the links actually are accessible.
        '''
        page = wikipedia.page(page_title, redirect= True, auto_suggest=False)
        all_links = page.links
        content = page.content
        permitted_links = []
        for i in all_links:
            if i.lower() in content.lower():
                permitted_links.append(i.lower())
        return permitted_links
    
    def test_path(self, arguments : dict) -> str:
        '''
        Reflexion test_path function, takes a dict like {"path": "Barack Obama -> Indonesia -> India"} and returns True if the path works, for a path like "Barack Obama -> Indonesia -> Pink Floyd" it would return "This path works until Pink floyd, which is not accessible from Indonesia".
        '''
        path = arguments["path"].split("->")
        if path[0].strip() != self.current_page.title:
            return "ERROR: The title of the start page of this path is not the title of your current page."
        for i in range(len(path)-1):
            if path[i+1].strip().lower() in self.get_links_of_page(path[i]):
                continue
            else:
                return "This path works until " + path[i+1].strip() + ", which is not accessible from " +path[i].strip()
        return "This path completely works."
    
    def get_links(self):
        '''
        Gets the list of permitted links from the current page. This is a different function than the get_links_of_page function because loading pages through the wikipedia api takes time, and we've already loaded the current page in, so there's no need to reload the page (whereas for the models suggested paths, we *have* to load a bunch of new pages in no matter what).
        '''
        all_links = self.current_page.links
        content = self.current_page.content
        permitted_links = []
        for i in all_links:
            if i.lower() in content.lower():
                permitted_links.append(i.lower())
        return permitted_links

    def is_permitted_link(self, link):
        '''
        Checks if a link is accessible from the current page. Checked when the agent calls move_page.
        '''
        if link.lower() in self.get_links():
            return True
        else:
            return False

    def get_title(self):
        '''
        Gets the title of the current page. Basically a completely unnecessary vestigial function, since you can just call self.current_page.title. It is called though sometimes so delete with care.
        '''
        return self.current_page.title

    def get_start(self):
        '''
        Gets the title of the starting page, similar to the above, totally unnecessary. It is called though sometimes so delete with care.
        '''
        return self.starting_page.title

    def get_goal(self):
        '''
        Gets the title of the goal page, also unnecessary. It is called though sometimes so delete with care.
        '''
        return self.goal_page.title
    def check_win(self):
        '''
        Checks if the game has been won by the agent
        '''
        if self.current_page==self.goal_page:
            return True
        else:
            return False

# a = WikipediaGame("Barack Obama", "India")
#%%

'''
Ensure that the wikipedia tool names correspond to callable functions on the WikipediaGame class (as this is how I'm going to handle them to avoid annoying things).

Below is a list of all the tools we're using.
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
                                "description": "The title of the new page you want to move to. This can be sensitive to plurals, or rewordings.",
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
                    "description" : "Accepts a test path string in the form \"current_page -> new_page1 -> new_page2 -> ... -> goal_page\" and returns where the path goes wrong if the path does not work, or \"success\" if the path works. Path titles can be sensitive to plurals or rewordings.",
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

*********************COMMENT************************

This was me messing around with something. Commented out as there's no need for it to run, but I also would rather not delete it right now because working out the regexes are kind of annoying, and it might be useful at some point down the line

****************************************************


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
#%%
class WikipediaRacingAgent:
    def __init__(self, wikigame, model = "gpt-4o", tools = WikipediaGameTools):
        self.game=wikigame
        self.state_count=0
        self.model = model
        self.default_system_message = {
                "role" : "system", 
                "content" : "You are a wikipedia-racing AI. Your goal is to reach " + self.game.get_goal() + ". Your current page is " + self.game.get_title() + ". You can determine which links are accessible as they will be wrapped in <link> </link> tags when you call get_content."
            }
        try:
            self.additional_rules_list = list(self.game.rules.keys()) # Loads in additional rules to tell the agent about, if they exist.
        except:
            self.additional_rules_list = []
        self.additional_rules_string= self.generate_additional_rules_string() # Generates a string to append to the prompt to let the agent know about the additional rules. Not currently handled or used, but don't delete as will probably add later (it is trivial to add this, but depends on whether the agent saturates performance).
        self.default_user_message={
                "role" : "user",
                "content" : "\n\nYou are playing the wikipedia game. You're starting on page " + self.game.get_title() + ". You're goal page is " + self.game.goal_page.title + " In case you're unsure, " + self.game.goal_page.title + " has the following summary: \n\n[Begin Goal Page Summary]\n" + self.game.get_page_summary(self.game.goal_page) + "\n[End Goal Page Summary]" + self.additional_rules_string
                }
        self.all_messages = [self.default_system_message, self.default_user_message]
        self.tools = tools
        self.response=None
        self.messages = [self.default_system_message, self.default_user_message]
    
    def generate_additional_rules_string(self):
        '''
        Generates a string to append to the prompt to let the agent know about the additional rules. Not currently handled or used, but don't delete as will probably add later (it is trivial to add this, but depends on whether the agent saturates performance).
        '''
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
        '''
        The basic generate response function. Gets an agent output with the prompt, with access to tools. This is what is used in the MVP (with tools=True). In V2 it's only used in the context of the ReAct_framework_generate_response function.
        '''
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
    
    @retry_with_exponential_backoff
    def ReAct_framework_generate_response(self, self_evaluate=True):
        '''
        ReAct framework response function. Gets the agent to reason without tools. Then gets it to use the tools. Probably worth including information about the tools in the thought thing here.
        '''
        num_thoughts = 1
        num_actions = 1
        response_list=[]
        for i in range(num_thoughts+num_actions):
            if i%2 == 0:
                self.messages.append(user_message(
                    "Think about what you want to do."
                ))
                new_response = self.generate_response(tools=False)
                response_list.append(new_response)
                self.response = new_response
                self.messages.append(self.response)
                self.all_messages.append(self.response)
            else:
                self.messages.append(user_message(
                    "Now act on your thoughts."
                ))
                new_response = self.generate_response(tools=True)
                response_list.append(new_response)
                self.response = new_response
                self.messages.append(self.response)
                self.all_messages.append(self.response)
                return response_list
        
    def get_response(self):
        '''
        Returns the agent response.
        
        '''
        return self.response

    def do_tool_calls(self, output):
        '''
        Handles tool calling behaviour. If the agent calls tools, then we iterate through the list of tool calls. We use getattr to just call the tool directly from the game class. We have to handle some tools carefully (e.g. arguments are "None" for the get_content, but OpenAI forces us to use arguments ugh, move_page also needs to be handled carefully as it triggers new_state())
        
        '''
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
        '''
        When the agent moves page we trigger this function.

        high_state_string was a hacky thing to warn it to pay attention to what it's been doing in a vain attempt to imrpove performance. Could arguably be deleted and probably won't lose much.

        Need to update default user and system message to inform it of a new game, and then extend the list of messages with these.

        The bit at the end handles removing all the prior wikipedia content. It's pretty hacky and there's probably a better way to do it, but it works (if no wikipedia page opens with the character "[" at least, which I haven't yet encountered).
        '''
        if self.state_count>=4:
            high_state_string="\n\nYou should pay attention to the path you have taken so far. If you have been going around certain pages in circles, and you keep returning to the same page, then try a plan that will take you in a different direction."
        else:
            high_state_string=""
        if self.game.current_page.title in game.page_title_history[:-1]:
            already_visited_string = "\n\nNotice that you have *already* visited this page. You can see what went wrong last time you visited this page in the \"paths you've taken so far.\" Make sure you react to this appropriately, and don't just try the same thing you already tried before."
        else:
            already_visited_string=""
        self.default_system_message = {
                "role" : "system", 
                "content" : "You are a wikipedia-racing AI. Your goal is to reach " + self.game.get_goal() + ". Your current page is " + self.game.get_title() + ". You have access to 3 tools: \n\ntest_path: This takes as input a path forward, and tells you whether this path works, or if not, where it goes wrong.\nget_content: This gets the content of the wikipedia page you are currently on.\nmove_page: This takes you to a new page, which must be a page that is accessible from your current page."
            }
        self.default_user_message={
                "role" : "user",
                "content" : "[START GAME SUMMARY]\n\nYou are now on page: " + self.game.current_page.title + ". So far you have visisted the following list of pages in order:" +"".join(["\n" + str(i+1) + ". " + self.game.page_title_history[i] for i in range(len(self.game.page_title_history))]) + high_state_string + self.additional_rules_string + "\n[END GAME SUMMARY]\nMake sure to consider using the test_path tool to test your future plans. The paths can potentially be very long paths, and you might be able to gain a lot of information this way.\n"
                }
        self.messages.extend([self.default_system_message,self.default_user_message])

        for i in self.messages: 
            try: x = i.role
            except: x = i['role']
            if (x == "tool") and (i['content'][0]!="[") and (i['name'] == "get_content"):
                i['content'] = "[The content of the wikipedia page: " + self.game.page_title_history[-1] + "]"
        self.all_messages.extend([self.default_system_message,self.default_user_message])


# %%
'''

**************COMMENT*************

A list of candidate games to test the MVP and V2 agents on. We can do more AI focused stuff if you want, or come up with funnier/more interesting ideas here, but these seem to test the agent effectively.

**********************************

Games of varying difficulty:

Obama -> India
MVP results
V2 results

Colin McEvedy -> India
MVP results
V2 results

1511 Daléra -> John Hume
MVP results: 8
V2 results: 8

Obama -> Tsetse fly (the MVP agent can solve this in 18 steps (including start and end), this agent can do it in 12)
MVP results
V2 results

Doyle Brunson -> Montu
MVP results
V2 results

For to Next -> Mexichromis trilineata
MVP results: 
V2:

'''



game = WikipediaGame("Barack Obama", "Tsetse Fly")
agent = WikipediaRacingAgent(game, model = "gpt-4o")
#%%
'''


Agent loop. 

Prints a bunch of stuff so I can actually see what the agent is doing. The printing isn't perfect (especially with tool calls, as I can't return all the tool calls, because that would end the function, and there are sometimes more than one, could fix this with a list but haven't gotten around to it as it's a minor detail). IT generally should give a good idea of what the agent is seeing.


'''

def agent_loop():
    if game.check_win():
        print("Success")
        return ""

    responses = agent.ReAct_framework_generate_response()
    print("Response:", responses[0].content)
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
        new_state_string = ("-" * 50) + "\n\nNEW STATE: " + game.get_title() + "\n\n" + " -> ".join(game.page_title_history) + "\n\n" + ("-"*50)
        print(new_state_string)
        print("SYSTEM MESSAGE: " + agent.default_system_message["content"])
        print("USER MESSAGE: " + agent.default_user_message["content"])

#%%
'''
Loop to run the agent a fixed number of times. DO NOT "while" loop this ever as it costs lots of money !!!!!!!!!!!!!!
'''
for i in range(0,20):
    agent_loop()






#%%
print(" -> ".join(game.page_title_history[:-1]))


#%%
for i in agent.messages:
    try: x = i['content']
    except: x = i.content
    if x==None:
        for j in i.tool_calls:
            print(j.function.name," ", j.function.arguments)
    else:
        print(x)

#%%
print(game.is_permitted_link("Tamil Nadu"))
print(game.get_links())
print(game.current_page.content)



# %%
'''
test_game = WikipediaGame("Wildlife","Barack Obama")
#print(test_game.get_content({}))
print(test_game.test_path({"path" : "Wildlife -> intensive farming -> Insecticide -> Insect -> Tsetse Fly"}))

print(test_game.get_links())
print(test_game.current_page.links)
print("Intensive farming".lower() in test_game.current_page.content.lower())
'''
#%%

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