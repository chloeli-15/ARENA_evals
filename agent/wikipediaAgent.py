#%%
import torch as t
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
print(countrylist[1])

# %%
'''
wikipedia.page returns a page_object type with most useful properties being:
    - content: A string of all the text on the wikipedia article.
    - links: Contains all the links on the article page. (Lots of these links wouldn't be allowed in classic wikipedia racing, this is why I'm searching for those links that are also named in the Content section)

'''
current_page_title = "India"
current_page = wikipedia.page(current_page_title, auto_suggest = False, redirect = True)
print(current_page.title)

client = establish_client_OpenAI()

def get_page_summary(current_page_title: str):
    return wikipedia.summary(current_page_title, auto_suggest=False, redirect=True)

def get_page_content(current_page_title: str):
    #Outputs the content of the wikipedia page
    return add_links(wikipedia.page(current_page_title, auto_suggest=False, redirect=True).content,wikipedia.page(current_page_title, auto_suggest=False, redirect=True).links)

def add_links(current_page_content, links):
    for word in sorted(links, key=len, reverse=True):
        current_page_content = current_page_content.replace(" " + word + " ", " " + f"<link>{word}</link>" + " ")
    return current_page_content

# def get_all_page_links(current_page_title):
#     return wikipedia.page(current_page_title, auto_suggest = False, redirect = True).links

def get_page_links(current_page_title: str):
    #Our
    current_page = wikipedia.page(current_page_title, auto_suggest=False, redirect=True)
    all_links = current_page.links
    content = current_page.content
    permitted_links = []
    for i in all_links:
        if i in content:
            permitted_links.append(i)
    return permitted_links


def check_if_permitted_link(link: str, current_page_title: str):
    permitted_links = get_page_links(current_page_title)
    if link.lower() in np.char.lower(permitted_links):
        return True
    else:
        return False

#%%
page_object = wikipedia.page("Barack Obama", auto_suggest=False,redirect=True)
get_page_links("Barack Obama")
# content = page_object.content
# page_links = []
# for i in page_object.links:
#     if i in content:
#         page_links.append(i)
#     else:
#         pass
# print(page_links)
# print(page_object.content)
# print(len(get_page_links("India")))
# # print(len(get_all_page_links("India")))
# print(get_page_content("India"))
print(get_page_content("Barack Obama"))





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
            "name" : "move_to_new_page",
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

def new_message_get_content(current_page, tool_call):
    return {
        "tool_call_id" : tool_call.id,
        "role" : "tool",
        "name" : tool_call.function.name,
        "content" : get_page_content(current_page.title)
    }
def new_message_get_links(current_page, tool_call):
    return {
        "tool_call_id" : tool_call.id,
        "role" : "tool",
        "name" : tool_call.function.name,
        "content" : "; ".join(get_page_links(current_page.title))
    }


#%%
'''


START PAGE BELOW



'''

current_page=wikipedia.page("Sakhalin Oblast",auto_suggest=False,redirect=True)
'''


GOAL PAGE BELOW



'''

goal_page=wikipedia.page("Mike D'Orso")



'''


AGENT LOOP BELOW



'''








pageTitleList=[current_page.title]
def new_message_new_state(current_page, goal_page, pageTitleList,additional_rules=[]):
    if len(additional_rules) == 0:
        additional_rules_string= ""
    if len(additional_rules) == 1:
        additional_rules_string = "There is an additional rule: " + additional_rules[0]
    else:
        additional_rules_string = "There are some additional rules:"
        for i in additional_rules:
            additional_rules_string = additional_rules_string + " " + i + "."


    return [
            {
                "role" : "system", 
                "content" : "You are a wikipedia-racing AI. Your starting point was: " + pageTitleList[0] + ". Your goal is to reach " + goal_page.title + ". Your current page is " + current_page.title + ". The path you have taken so far is " + " -> ".join(pageTitleList) + "."
            },
            {
                "role" : "user",
                "content" : "You are currently on page: " + current_page.title + ". Before taking any actions, you should think about which articles on your current page will make it easier to get to " + goal_page.title + ". Don't plan your whole journey in advance, you never know what links might come up. In case you're unsure " + goal_page.title + " has the following summary: " + get_page_summary(goal_page.title) + "\nMake sure to read the content of the wikipedia page you are on, in order to choose the best possible link. Once you have chosen a link, you will not be able to go back to the current page, so choose the best option you can." + additional_rules_string
            }
        ]
@retry_with_exponential_backoff
def new_chat_completion(messages, model = "gpt-4o",tools = wikipedia_race_tools, tool_choice="auto"):
    return client.chat.completions.create(
        model = model,
        messages = messages,
        tools = tools,
        tool_choice = tool_choice
    )
additional_rules = []

messages=new_message_new_state(current_page,goal_page,pageTitleList,additional_rules)



list_of_messages = []


for i in range(0,40):
    output = new_chat_completion(messages, model = "gpt-4o")
    print(output.choices[0].message.content)
    messages.append(output.choices[0].message)
    tool_calls = output.choices[0].message.tool_calls
    if output.choices[0].message.tool_calls:
        for tool_call in tool_calls:
            if tool_call.function.name == "get_current_page_content":
                messages.append(new_message_get_content(current_page,tool_call))
                content = current_page.title + " Wikipedia Content"

            elif tool_call.function.name == "get_current_page_links":
                links = "; ".join(get_page_links(current_page.title)) #join them to make it a nicely presented string. Worth also trying comma separations, but thought that might get confusing since commas are more common in wikipedia titles.
                messages.append(new_message_get_links(current_page,tool_call))
                content = new_message_get_links(current_page,tool_call)['content']

            elif tool_call.function.name == "move_to_new_page":
                new_link=json.loads(tool_call.function.arguments)['new_page']
                if check_if_permitted_link(new_link, current_page.title) and new_link not in pageTitleList:
                    current_page = wikipedia.page(new_link, auto_suggest=False, redirect=True)
                    pageTitleList.append(current_page.title)
                    list_of_messages.extend(messages)
                    messages=new_message_new_state(current_page,goal_page,pageTitleList,additional_rules)
                    content = messages[0]['content']
                    content = "-------------------\n\n New State begins \n + " "Path so far: " + " -> ".join(pageTitleList) + "\n ------------------- \n\n" + "SYSTEM: " + content +"\n\n" + messages[1]['content']

                elif new_link in pageTitleList:
                    content = "You've already visited that page. Try going somewhere else."
                    messages.append(
                        {
                        "tool_call_id" : tool_call.id,
                        "role" : "tool",
                        "name" : tool_call.function.name,
                        "content" : content
                    }
                    )
                elif new_link in countrylist:
                    content = "That's not allowed according to the additional rules."
                    messages.append(
                        {
                        "tool_call_id" : tool_call.id,
                        "role" : "tool",
                        "name" : tool_call.function.name,
                        "content" : content
                    }
                    )
                else:
                    content = "That link was not valid. You'll have to try something else."
                    messages.append(
                        {
                        "tool_call_id" : tool_call.id,
                        "role" : "tool",
                        "name" : tool_call.function.name,
                        "content" : content
                    }
                    )
            messages.append(
                        {
                            "role" : "user",
                            "content" : "What's your next step?"
                        }
                    )
    if tool_calls != None:
        print(tool_calls[0].function.name)
    print(content)
    print("What's your next step?")
    if current_page==goal_page:
        print("Success")
        break
    output=""

#%%



print(json.loads(tool_call.function.arguments))
for i in list_of_messages:
    try: print(i['content'])
    except: print(i.content)

    #%%
print(len(messages))
'''
print(messages[-4])
print(messages[-3])
print(messages[-2])
print(messages[-1])
'''
print(" -> ".join(pageTitleList))
for i in messages:
        print(i)




#%%
