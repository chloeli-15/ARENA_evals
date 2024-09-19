#%%
#Trying to work out the password game agent. Design + Coming up with the Game + Tooling
import torch as t
from openai import OpenAI
from pprint import pprint
import string

class Password:
    '''
    Making the password a class because its easier to centralize all of this into one data type probably
    '''

    def __init__(self, current_password):
        self.current_password = current_password #Initiate password

        #Do rule checking:
        self.contains_five_chars = self.contains_five_chars_rule()
        self.contains_number = self.contains_number_rule()
        self.contains_uppercase = self.contains_uppercase_rule()
        self.contains_special_character=self.contains_special_character_rule()
        self.digit_sum_to_25 = self.digit_sum_to_25_rule()
        self.contains_month = self.contains_month_rule()



    def edit(self, new_password):
        self.current_password = new_password #Change password

        #Do rule checking on new password and update attributes:
        self.contains_five_chars = self.contains_five_chars_rule()
        self.contains_number = self.contains_number_rule()
        self.contains_uppercase = self.contains_uppercase_rule()
        self.contains_special_character=self.contains_special_character_rule()
        self.digit_sum_to_25 = self.digit_sum_to_25_rule()
        self.contains_month = self.contains_month_rule()



    def contains_five_chars_rule(self):
        if len(self.current_password)<5:
            return False
        else:
            return True
    def contains_number_rule(self):
        for i in string.digits:
            if i in self.current_password:
                return True
        return False
    def contains_uppercase_rule(self):
        for i in string.ascii_uppercase:
            if i in self.current_password:
                return True
        return False
    def contains_special_character_rule(self):
        for i in string.punctuation:
            if i in self.current_password:
                return True
        return False
    def digit_sum_to_25_rule(self):
        numbers_list=[]
        for i in self.current_password:
            if i in string.digits:
                numbers_list.append(int(i))
        if sum(numbers_list)==25:
            return True
        return False
    def contains_month_rule(self):
        monthList = ["january", "february", "march", "april", "may", "june", "july", "august", "september","october","november","december"]
        for i in monthList:
            if i in self.current_password.lower():
                return True
        return False
    def contains_roman_numeral(self):
        roman_numeral_list=["I","V","X","L","C","D","M"]
        for i in roman_numeral_list:
            if i in self.current_password:
                return True
        return False
    
    




x = Password("hello")
x.edit("Hello")

print(x.current_password + "\n")
print("contains_five_chars: " + str(x.contains_five_chars))
print("contains_uppercase: " + str(x.contains_uppercase))
print("contains_special_character: " + str(x.contains_special_character))
print("digit_sum_to_25: " + str(x.digit_sum_to_25))

x.edit("Hello!5555.05")
print(x.current_password + "\n")
print("contains_special_character: " + str(x.contains_special_character))
print("digit_sum_to_25: " + str(x.digit_sum_to_25))

# %%
