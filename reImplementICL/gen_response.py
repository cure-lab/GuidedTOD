import argparse
import os
import random
import jsonlines
from openai import OpenAI
import json
from metrics import compute_ast_acc_metrics_abcd, compute_ast_acc_metrics_multiwoz
from tqdm import tqdm
import time

# set seed
random.seed(42)

clientGPT4 = OpenAI(api_key="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")
clientGPT3_5 = OpenAI(api_key="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")

def call_LLM_woaction_abcd(dialogue, Action=None, gpt_model="gpt-3.5-turbo"):
    action_description = {
        "pull-up-account": "account has been pulled up for <name>.",
        "enter-details": "details of <username> have been entered.",
        "verify-identity": "identity verification in progress ...",
        "make-password": "a password has been generated.",
        "search-timing": "system action: search timing, I need to ask a certain question about timing.",
        "search-policy": "system action: search policy, what kind of policy does the customer want to know?",
        "validate-purchase": "purchase validation in progress ...",
        "search-faq": "Answers can be found in the faq pages, searching the faq pages ...",
        "membership": "membership level of <level> has been noted.",
        "search-boots": "system action: search boots, click the boots toggle switch",
        "try-again": "agent is looking for solutions ...",
        "ask-the-oracle": "querying the system for an answer ...",
        "update-order": "order has been updated with <change>.",
        "promo-code": "a promo code has been created.",
        "update-account": "account has been updated with <change>.",
        "search-membership": "system action: search membership, I need to know the membership level of the customer.",
        "make-purchase": "a purchase of <item> was made.",
        "offer-refund": "a refund has been made for the amount of $<amount>.",
        "notify-team": "the website team has been notified.",
        "record-reason": "a reason of <reason> has been recorded.",
        "search-jeans": "system action: search jeans, click the jeans toggle switch",
        "shipping-status": "shipping status of <status> has been noted.",
        "search-shirt": "system action: search shirt, click the shirt toggle switch",
        "instructions": "agent is looking for solutions ..., I will give you some instructions.",
        "search-jacket": "system action: search jacket, click the jecket toggle switch",
        "log-out-in": "agent is looking for solutions ..., instruct the customer to log out of their account and log back in.",
        "select-faq": "faq answer related to <faq> was selected.",
        "subscription-status": "querying the system for subscription status ...",
        "send-link": "a link will be sent.",
        "search-pricing": "system action: search pricing, price of something."
    }
    abcd_hint_prompt = """
    The following are conversations between a user and an assistant. Indicated by the dialog acts, the assistant can help the user with checking in or providing information of temperature, time, price, location, and so on.
    You should predict the next action the assistant should take based on the context of the conversation.
    The action should be taken from the list of dialog acts provided below.
    Also, you need to fill in the slot value along with the action, if any, if no slot value is required, you should make the slot value be none. The format is action_name [none].

    Available Dialog acts:
    - pull-up-account: account has been pulled up for <name>.
    - enter-details: details of <username> have been entered.
    - verify-identity: identity verification in progress ...
    - make-password: a password has been generated.
    - search-timing: system action: search timing, I need to ask a certain question about timing.
    - search-policy: system action: search policy, what kind of policy does the customer want to know?
    - validate-purchase: purchase validation in progress ...
    - search-faq: Answers can be found in the faq pages, searching the faq pages ...
    - membership: membership level of <level> has been noted.
    - search-boots: system action: search boots, click the boots toggle switch
    - try-again: agent is looking for solutions ...
    - ask-the-oracle: querying the system for an answer ...
    - update-order: order has been updated with <change>.
    - promo-code: a promo code has been created.
    - update-account: account has been updated with <change>.
    - search-membership: system action: search membership, I need to know the membership level of the customer.
    - make-purchase: a purchase of <item> was made.
    - offer-refund: a refund has been made for the amount of $<amount>.
    - notify-team: the website team has been notified.
    - record-reason: a reason of <reason> has been recorded.
    - search-jeans: system action: search jeans, click the jeans toggle switch
    - shipping-status: shipping status of <status> has been noted.
    - search-shirt: system action: search shirt, click the shirt toggle switch
    - instructions: agent is looking for solutions ..., I will give you some instructions.
    - search-jacket: system action: search jacket, click the jecket toggle switch
    - log-out-in: agent is looking for solutions ..., instruct the customer to log out of their account and log back in.
    - select-faq: faq answer related to <faq> was selected.
    - subscription-status: querying the system for subscription status ...
    - send-link: a link will be sent.
    - search-pricing: system action: search pricing, price of something.

    Conversation: 
    Context: hello, how may i help you? i want to know the state of my refund. let me help you with that. i have an existing refund of $100 + i want to refund another $<amount>. did you want to add an extra item to your current refund? yes. could i have your full name or account id? albert sanders. account id 123445.
    Assistant: pull-up-account [albert sanders]

    Conversation:
    Context: Context: hello, how may i help you? i want to know the state of my refund. let me help you with that. i have an existing refund of $100 + i want to refund another $<amount>. did you want to add an extra item to your current refund? yes. could i have your full name or account id? albert sanders. account id 123445. thanks. could i have your username, email address and order id to validate your order? <username>. <email>. and the order id? <order_id>. thank you. what is the  item that you want to return? jeans. <name>. 
    Assistant: record-reason [guess jeans]

    Conversation:
    Context: hi. i want to manage my shipping details as my situation has changed. welcome to acmebrands! how may i help you today? i see. what is your name please? i want to change my shipping address. rodriguez domingo. and what is the shipping status please? order received. thanks.
    Assistant: shipping-status [order received]

    Conversation:
    Context: i would like to know more about a product. hello. how may i help you today? sure. i would like to know if the buttons are brown or black. i see. so you are looking to purchase buttons? is there a drop down menu to select the color buttons you want to buy? no im looking to buy a shirt and asking if the button on the shirt is brown or black. product: shirt  brand: michael_kors  amount: $<amount>. oh the buttons on a shirt? should have mentioned that at the beginning.  let me take a look for you. that shirt has dark brown buttons on them. 
    Assistant: select-faq [shirt_other_3]

    Conversation: 
    Context: hi! how may i help you? hello. i recently signed up for a subscription but it looks like you guys charged me twice for it. i see, let's fix that. may i have your full name, account and order ids? sure, it's albert sanders and my account id is <account_id> do you have an order id? yes its <order_id>
    Assistant: verify-identity [albert sanders, <account_id>, <account_id>]

    Conversation:
    Context: hello, thank you for contacting us today. how can i help you? how do you cancel a subscription? i'm sorry to hear that you might want to cancel your subscription. did something happen that made you want to do this? no, not at all.  i was just thinking of ordering some things and i don't want to if the cancelation process is too hard. alright let me see what i can find for you.
    Assistant: search-policy [none]

    Conversation: 
    [[DIALOG]]
    """
    abcd_hint_prompt = abcd_hint_prompt.strip()

    prompt = abcd_hint_prompt.replace("[[DIALOG]]", dialogue)

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant. You can predict the next action given the preivous utterance context."})
    messages.append({"role": "user", "content": prompt})

    response = clientGPT4.chat.completions.create(
        # model="gpt-3.5-turbo",
        model=gpt_model,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response.choices[0].message.content)

    return response.choices[0].message.content

def call_LLM_waction_abcd(dialogue, Action=None, gpt_model="gpt-3.5-turbo"):
    action_description = {
        "pull-up-account": "account has been pulled up for <name>.",
        "enter-details": "details of <username> have been entered.",
        "verify-identity": "identity verification in progress ...",
        "make-password": "a password has been generated.",
        "search-timing": "system action: search timing, I need to ask a certain question about timing.",
        "search-policy": "system action: search policy, what kind of policy does the customer want to know?",
        "validate-purchase": "purchase validation in progress ...",
        "search-faq": "Answers can be found in the faq pages, searching the faq pages ...",
        "membership": "membership level of <level> has been noted.",
        "search-boots": "system action: search boots, click the boots toggle switch",
        "try-again": "agent is looking for solutions ...",
        "ask-the-oracle": "querying the system for an answer ...",
        "update-order": "order has been updated with <change>.",
        "promo-code": "a promo code has been created.",
        "update-account": "account has been updated with <change>.",
        "search-membership": "system action: search membership, I need to know the membership level of the customer.",
        "make-purchase": "a purchase of <item> was made.",
        "offer-refund": "a refund has been made for the amount of $<amount>.",
        "notify-team": "the website team has been notified.",
        "record-reason": "a reason of <reason> has been recorded.",
        "search-jeans": "system action: search jeans, click the jeans toggle switch",
        "shipping-status": "shipping status of <status> has been noted.",
        "search-shirt": "system action: search shirt, click the shirt toggle switch",
        "instructions": "agent is looking for solutions ..., I will give you some instructions.",
        "search-jacket": "system action: search jacket, click the jecket toggle switch",
        "log-out-in": "agent is looking for solutions ..., instruct the customer to log out of their account and log back in.",
        "select-faq": "faq answer related to <faq> was selected.",
        "subscription-status": "querying the system for subscription status ...",
        "send-link": "a link will be sent.",
        "search-pricing": "system action: search pricing, price of something."
    }
    abcd_hint_prompt = """
    The following are conversations between a user and an assistant. Indicated by the dialog acts, the assistant can help the user with checking in or providing information of temperature, time, price, location, and so on.
    You should predict the next action the assistant should take based on the context of the conversation.
    The action should be taken from the list of dialog acts provided below.
    Also, you need to fill in the slot value along with the action, if any, if no slot value is required, you should make the slot value be none. The format is action_name [none].
    Previous actions have been added into the context for each conversation.

    Available Dialog acts:
    - pull-up-account: account has been pulled up for <name>.
    - enter-details: details of <username> have been entered.
    - verify-identity: identity verification in progress ...
    - make-password: a password has been generated.
    - search-timing: system action: search timing, I need to ask a certain question about timing.
    - search-policy: system action: search policy, what kind of policy does the customer want to know?
    - validate-purchase: purchase validation in progress ...
    - search-faq: Answers can be found in the faq pages, searching the faq pages ...
    - membership: membership level of <level> has been noted.
    - search-boots: system action: search boots, click the boots toggle switch
    - try-again: agent is looking for solutions ...
    - ask-the-oracle: querying the system for an answer ...
    - update-order: order has been updated with <change>.
    - promo-code: a promo code has been created.
    - update-account: account has been updated with <change>.
    - search-membership: system action: search membership, I need to know the membership level of the customer.
    - make-purchase: a purchase of <item> was made.
    - offer-refund: a refund has been made for the amount of $<amount>.
    - notify-team: the website team has been notified.
    - record-reason: a reason of <reason> has been recorded.
    - search-jeans: system action: search jeans, click the jeans toggle switch
    - shipping-status: shipping status of <status> has been noted.
    - search-shirt: system action: search shirt, click the shirt toggle switch
    - instructions: agent is looking for solutions ..., I will give you some instructions.
    - search-jacket: system action: search jacket, click the jecket toggle switch
    - log-out-in: agent is looking for solutions ..., instruct the customer to log out of their account and log back in.
    - select-faq: faq answer related to <faq> was selected.
    - subscription-status: querying the system for subscription status ...
    - send-link: a link will be sent.
    - search-pricing: system action: search pricing, price of something.

    Conversation: 
    Context: hello, how may i help you? i want to know the state of my refund let me help you with that. i have an existing refund of $100 + i want to refund another $<amount> did you want to add an extra item to your current refund? yes could i have your full name or account id? albert sanders account id 123445 pull-up-account ['albert sanders'] thanks could i have your username, email address and order id to validate your order? <username> <email> and the order id? <order_id>
    Assistant: validate-purchase [<username>, <username>, <username>]

    Conversation:
    Context: hello, how may i help you? i want to know the state of my refund let me help you with that. i have an existing refund of $100 + i want to refund another $<amount> did you want to add an extra item to your current refund? yes could i have your full name or account id? albert sanders account id 123445
    Assistant: pull-up-account [albert sanders]

    Conversation:
    Context: hi i want to manage my shipping details as my situation has changed welcome to acmebrands! how may i help you today? i see. what is your name please? i want to change my shipping address rodriguez domingo pull-up-account ['rodriguez domingo'] and what is the shipping status please? order received thanks shipping-status ['order received'] next, i need your username, email & order id username: <username> email address: <email> order id: <order_id>
    Assistant: validate-purchase [<username>, <username>, <username>]

    Conversation:
    Context: hello thanks for contacting acmebrands, how can i help you? hello! i figuring out to use the hood of the jacket which jacket are you asking about? a michales kors jacket. how can i detach the hood?
    Assistant: search-faq [none]

    Conversation: 
    Context: hello how are you doing today i need some help regarding my account hello what is the issue with your account? i ordered some clothing items and put in the wrong address you want to change your address now? yes please i currently moved and put in my old shipping address i need to put in my new address please give me your full name crystal minh pull-up-account ['crystal minh'] what is the current status of the order? it says out for delivery shipping-status ['out for delivery'] now i need to validate your purchase give me your username, email and order id my user name is <username> my email address is <email> my order id is <order_id> validate-purchase ['<username>', '<username>', '<username>'] verified! thank you so much
    Assistant: update-order [change order]

    Conversation:
    Context: good afternoon, what can i do for you? hi, i can't access my account because i forgot my password. i understand, could you give me your full name or account id please sure. my name is norman bouchard. pull-up-account ['norman bouchard'] thanks, what is your username? norbou
    Assistant: enter-details [norbou]

    Conversation: 
    [[DIALOG]]
    """
    abcd_hint_prompt = abcd_hint_prompt.strip()

    prompt = abcd_hint_prompt.replace("[[DIALOG]]", dialogue)

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant. You can predict the next action given the preivous utterance context."})
    messages.append({"role": "user", "content": prompt})

    response = clientGPT4.chat.completions.create(
        # model="gpt-3.5-turbo",
        model=gpt_model,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response.choices[0].message.content)

    return response.choices[0].message.content

def call_LLM_woaction_multiwoz(dialogue, Action=None, gpt_model="gpt-3.5-turbo"):
    action_description = {
        "find_hotel": "customers are looking for hotels with specific requirements",
        "book_hotel": "customers are going to booking hotels",
        "find_train": "customers are looking for trains with specific requirements",
        "book_train": "customers are going to booking train tickets",
        "find_attraction": "customers are looking for attractions with specific requirements",
        "find_restaurant": "customers are looking for restaurants with specific requirements",
        "book_restaurant": "customers are going to booking tables at restaurants",
        "find_hospital": "customers are looking for hospitals with specific requirements",
        "book_taxi": "customers are going to booking taxis",
        "find_taxi": "customers are looking for taxis with specific requirements",
        "find_bus": "customers are looking for buses with specific requirements",
        "find_police": "customers are looking for police stations"
    }

    abcd_hint_prompt = """
    The following are conversations between a user and an assistant. Indicated by the dialog acts, the assistant can help the user with checking in or providing information of temperature, time, price, location, and so on.
    You should predict the next action the assistant should take based on the context of the conversation.
    The action should be taken from the list of dialog acts provided below.
    Also, you need to fill in the slot value along with the action, if any, if no slot value is required, you should make the slot value be none. The format is action name [none].

    Available Dialog acts:
    - search for hotel: customers are looking for hotels with specific requirements
    - book hotel: customers are going to booking hotels
    - search for trains: customers are looking for trains with specific requirements
    - book train ticket: customers are going to booking train tickets
    - search for attractions: customers are looking for attractions with specific requirements
    - search for restaurants: customers are looking for restaurants with specific requirements
    - book table at restaurant: customers are going to booking tables at restaurants
    - search for hospital: customers are looking for hospitals with specific requirements
    - book taxi: customers are going to booking taxis
    - search for taxi: customers are looking for taxis with specific requirements
    - search for bus: customers are looking for buses with specific requirements
    - search for police station: customers are looking for police stations

    Conversation: 
    Context: i need a list of cheap place -s to stay that include free parking . alexander bed and breakfast is in the cheap price range in the centre of town . okay , does that place include free wifi and it is 4 stars ? yes , the alexander has free wifi and is a 4 star hotel . how many nights will you be staying ? i will be staying 5 nights starting from saturday .
    Assistant: search for hotel [with parking, cheap, with internet, alexander bed and breakfast, 4 stars]

    Conversation:
    Context: can you let me know if a place called the gonville hotel is still around ? yes and it is as popular as ever . it is 3 stars and quite expensive . would you like me to book a room for you ? what area of town is it in ? it s in town centre . ok , thanks . also , are there any indian restaurant -s in the centre ?
    Assistant: search for restaurants [centre, indian]

    Conversation:
    Context: i am looking for a train leaving on saturday from kings lynn . do you have any time preferences ? i need to leave after 13:00 . what will your destination be ? i am wanting to go to cambridge . the tr1499 leaves at 17:11 . can i book some tickets for you ? yes book for 4 people you are reference number is biazmbuc . is there anything else i can help you with ? i am looking for an expensive restaurant in the centre city . i have several restaurant -s in the centre in the expensive range . what type of cuisine would your prefer ? no preference . can you recommend 1 & book a table for 4 people at 17:45 on saturday . & may i have the reference # please ? i would recommend british cuisine . would that be okay with you ? that would be fine . is it available saturday at 17:45 ? the restaurant fitzbillies is available , and i have made you a reservation . your reference number is 4wgdgosa . thank you for your help , have a nice day .
    Assistant: book table at restaurant [saturday, 4 people, 17:45]

    Conversation:
    Context: hi , i am looking to visit the colleges in town , and need to know the names of some of them . there are 18 matches . do you have an area in mind ? are there any in the centre ? there are 13 in the centre , including christ s college and emmanuel college . the area actually does not really matter , i just need the address of 1 of your recommendations , please ? sure ! the address for christ s college is saint andrew s street , postcode cb23bu . thank you ! i am also looking for a hotel with 4 stars . the university arms hotel is 4 starts and in the centre where the college is , would you like me to book you or provide more information ? do they have internet and could i have the phone number please ? yes they have internet and phone 01223351241 i need to book a taxi also . it should leave the hotel by 11:30
    Assistant: search for taxi [university arms hotel, christ s college, 11:30]

    Conversation: 
    Context: i am looking for a train that will depart from birmingham new street and go to cambridge . can you help me ? there are 133 trains from birmingham new street and to cambridge . is there a particular day and and or time you are interested in ? yes , i want to leave on monday and i need to arrive by 11:15 train tr5747 departs at 7:40 and would arrive in cambridge at 10:23 . would that work for you ? yes , that would be perfect , thanks . here is your reference number: qqvpn13m . total fee will be 75.09 gbp payable at the station . is there anything else i can help you with today ? can you help me find a place to stay ? i am looking a 0 star hotel with a cheap price . i found 2 zero star guest houses that are cheap . 1 is in the north and 1 in the centre . which area would you prefer ? are you able to find me a hotel in the east with free parking ? there are 3 cheap guest houses in east with free parking . they are all 4 stars . would you like me to book 1 of those ? no , i would like that zero star cheap guest house in the north . sure , i can help you with that . when would you like to stay ? monday . please give me their phone number , postcode and address
    Assistant: book train ticket [none]

    Conversation:
    Context: hi , i am looking for the nearest police station . hi , the nearest police station is at parkside , cambridge postcode: cb11jg . is there anything else i can help with ? can you please give me the phone number as well ?
    Assistant: search for police station [none]

    Conversation: 
    [[DIALOG]]
    """
    abcd_hint_prompt = abcd_hint_prompt.strip()

    prompt = abcd_hint_prompt.replace("[[DIALOG]]", dialogue)

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant. You can predict the next action given the preivous utterance context."})
    messages.append({"role": "user", "content": prompt})

    response = clientGPT4.chat.completions.create(
        # model="gpt-4"
        model=gpt_model,
        # model="gpt-4-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response.choices[0].message.content)

    return response.choices[0].message.content

def call_LLM_waction_multiwoz(dialogue, Action=None, gpt_model="gpt-3.5-turbo"):
    action_description = {
        "find_hotel": "customers are looking for hotels with specific requirements",
        "book_hotel": "customers are going to booking hotels",
        "find_train": "customers are looking for trains with specific requirements",
        "book_train": "customers are going to booking train tickets",
        "find_attraction": "customers are looking for attractions with specific requirements",
        "find_restaurant": "customers are looking for restaurants with specific requirements",
        "book_restaurant": "customers are going to booking tables at restaurants",
        "find_hospital": "customers are looking for hospitals with specific requirements",
        "book_taxi": "customers are going to booking taxis",
        "find_taxi": "customers are looking for taxis with specific requirements",
        "find_bus": "customers are looking for buses with specific requirements",
        "find_police": "customers are looking for police stations"
    }
    abcd_hint_prompt = """
    The following are conversations between a user and an assistant. Indicated by the dialog acts, the assistant can help the user with checking in or providing information of temperature, time, price, location, and so on.
    You should predict the next action the assistant should take based on the context of the conversation.
    The action should be taken from the list of dialog acts provided below.
    Also, you need to fill in the slot value along with the action, if any, if no slot value is required, you should make the slot value be none. The format is action name [none].
    Previous actions have been added into the context for each conversation.

    Available Dialog acts:
    - search for hotel: customers are looking for hotels with specific requirements
    - book hotel: customers are going to booking hotels
    - search for trains: customers are looking for trains with specific requirements
    - book train ticket: customers are going to booking train tickets
    - search for attractions: customers are looking for attractions with specific requirements
    - search for restaurants: customers are looking for restaurants with specific requirements
    - book table at restaurant: customers are going to booking tables at restaurants
    - search for hospital: customers are looking for hospitals with specific requirements
    - book taxi: customers are going to booking taxis
    - search for taxi: customers are looking for taxis with specific requirements
    - search for bus: customers are looking for buses with specific requirements
    - search for police station: customers are looking for police stations

    Conversation: 
    Context: hello, how may i help you? i want to know the state of my refund let me help you with that. i have an existing refund of $100 + i want to refund another $<amount> did you want to add an extra item to your current refund? yes could i have your full name or account id? albert sanders account id 123445 pull-up-account ['albert sanders'] thanks could i have your username, email address and order id to validate your order? <username> <email> and the order id? <order_id>
    Assistant: validate-purchase [<username>, <username>, <username>]

    Conversation:
    Context: hello, how may i help you? i want to know the state of my refund let me help you with that. i have an existing refund of $100 + i want to refund another $<amount> did you want to add an extra item to your current refund? yes could i have your full name or account id? albert sanders account id 123445
    Assistant: pull-up-account [albert sanders]

    Conversation:
    Context: hi i want to manage my shipping details as my situation has changed welcome to acmebrands! how may i help you today? i see. what is your name please? i want to change my shipping address rodriguez domingo pull-up-account ['rodriguez domingo'] and what is the shipping status please? order received thanks shipping-status ['order received'] next, i need your username, email & order id username: <username> email address: <email> order id: <order_id>
    Assistant: validate-purchase [<username>, <username>, <username>]

    Conversation:
    Context: hello thanks for contacting acmebrands, how can i help you? hello! i figuring out to use the hood of the jacket which jacket are you asking about? a michales kors jacket. how can i detach the hood?
    Assistant: search-faq [none]

    Conversation: 
    Context: hello how are you doing today i need some help regarding my account hello what is the issue with your account? i ordered some clothing items and put in the wrong address you want to change your address now? yes please i currently moved and put in my old shipping address i need to put in my new address please give me your full name crystal minh pull-up-account ['crystal minh'] what is the current status of the order? it says out for delivery shipping-status ['out for delivery'] now i need to validate your purchase give me your username, email and order id my user name is <username> my email address is <email> my order id is <order_id> validate-purchase ['<username>', '<username>', '<username>'] verified! thank you so much
    Assistant: update-order [change order]

    Conversation:
    Context: good afternoon, what can i do for you? hi, i can't access my account because i forgot my password. i understand, could you give me your full name or account id please sure. my name is norman bouchard. pull-up-account ['norman bouchard'] thanks, what is your username? norbou
    Assistant: enter-details [norbou]

    Conversation: 
    [[DIALOG]]
    """
    abcd_hint_prompt = abcd_hint_prompt.strip()

    prompt = abcd_hint_prompt.replace("[[DIALOG]]", dialogue)

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant. You can predict the next action given the preivous utterance context."})
    messages.append({"role": "user", "content": prompt})

    response = clientGPT4.chat.completions.create(
        # model="gpt-3.5-turbo",
        model=gpt_model,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response.choices[0].message.content)

    return response.choices[0].message.content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="abcd")
    parser.add_argument('--waction', action="store_true", help="with or without actions")
    parser.add_argument('--test_file', type=str, default="/research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/data/processed/test_AST_abcd_woaction_flow_all.json")
    parser.add_argument('--file_suffix', type=str, default="responses/abcd_woaction.json")
    parser.add_argument('--gpt_model', type=str, default="gpt-3.5-turbo")

    args = parser.parse_args()
    print(args)

    convo_ids = []
    turn_counts = []
    contexts = []
    labels = []
    if args.dataset == "abcd":
        flows = []
    predictions = []
    save_path = f"responses/{args.dataset}_waction_{args.gpt_model}_{args.file_suffix}.json" if args.waction else f"responses/{args.dataset}_woaction_{args.gpt_model}_{args.file_suffix}.json"

    if os.path.exists(save_path):
        os.remove(save_path)

    with jsonlines.open(args.test_file) as reader:
        for sample in reader:
            convo_ids.append(sample["convo_id"])
            turn_counts.append(sample["turn_id"])
            contexts.append(sample["input"])
            if args.dataset == "abcd":
                labels.append(sample["target"].split(":")[1].strip())
                flows.append(sample["flow"])
            elif args.dataset == "multiwoz":
                labels.append(sample["target"])
            else:
                print("invalid experiment name")
    
    # randomly select 500 samples to save money
    if args.gpt_model == "gpt-4":
        indices = random.sample(range(len(contexts)), 500)
        contexts = [contexts[i] for i in indices]
        labels = [labels[i] for i in indices]
        convo_ids = [convo_ids[i] for i in indices]
        turn_counts = [turn_counts[i] for i in indices]
        if args.dataset == "abcd":
            flows = [flows[i] for i in indices]
    

    for i, context in tqdm(enumerate(contexts), total=len(contexts)):
        if args.dataset == "multiwoz":
            if "Possible Actions: []" in context:
                context = context.replace("Possible Actions: []", "").strip()
        dialogue = f"{context}\nAssistant: "
        label = labels[i]
        for try_time in range(3):
            try:
                if args.waction:
                    if args.dataset == "abcd":
                        response = call_LLM_waction_abcd(dialogue, label, gpt_model=args.gpt_model)
                    elif args.dataset == "multiwoz":
                        response = call_LLM_waction_multiwoz(dialogue, label, gpt_model=args.gpt_model)
                    else:
                        print("invalid experiment name")
                else:
                    if args.dataset == "abcd":
                        response = call_LLM_woaction_abcd(dialogue, label, gpt_model=args.gpt_model)
                    elif args.dataset == "multiwoz":
                        response = call_LLM_woaction_multiwoz(dialogue, label, gpt_model=args.gpt_model)
                    else:
                        print("invalid experiment name")
                break
            except:
                if try_time == 2:
                    response = "MISSING"
                    print("cannot get response from LLM")
                print("retrying ...")
                time.sleep(1)
                continue
        
        predictions.append(response)

        save_context = {"sample_id": i, "convo_id": convo_ids[i], "turn_id": turn_counts[i], "target": label, "input": context, "pred_action": response}
        
        with open(save_path, "a") as w:
            json.dump(save_context, w)
            w.write("\n")

        if i % 20 == 0:
            print(f"label: {label}, response: {response}")
            

    sequence_scores = [1 for _ in range(len(predictions))]
    if args.dataset == "abcd":
        results = compute_ast_acc_metrics_abcd(predictions, labels, convo_ids, turn_counts, sequence_scores=sequence_scores, num_beams=1)
    elif args.dataset == "multiwoz":
        results = compute_ast_acc_metrics_multiwoz(predictions, labels, convo_ids, turn_counts, sequence_scores=sequence_scores, num_beams=1)
    print(json.dumps(results, indent=4))