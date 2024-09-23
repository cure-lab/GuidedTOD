import json
from openai import OpenAI
import time
import argparse
import jsonlines
from tqdm import tqdm
import os

clientGPT = OpenAI(api_key="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")
action_description = {
    "find_hotel": "customers are looking for hotels with",
    "book_hotel": "customers are going to booking hotels for",
    "find_train": "customers are looking for trains from",
    "book_train": "customers are going to booking train tickets for",
    "find_attraction": "customers are looking for attractions of",
    "find_restaurant": "customers are looking for restaurants of",
    "book_restaurant": "customers are going to booking tables at restaurants for",
    "find_hospital": "customers are looking for hospitals in",
    "book_taxi": "customers are going to booking taxis",
    "find_taxi": "customers are looking for a taxi with at",
    "find_bus": "customers are looking for a bus from",
    "find_police": "customers are looking for police stations"
}

action_mapping = {
    "find_hotel": "search for hotel",
    "book_hotel": "book hotel",
    "find_train": "search for trains",
    "book_train": "book train ticket",
    "find_attraction": "search for attractions",
    "find_restaurant": "search for restaurants",
    "book_restaurant": "book table at restaurant",
    "find_hospital": "search for hospital",
    "book_taxi": "book taxi",
    "find_taxi": "search for taxi",
    "find_bus": "search for bus",
    "find_police": "search for police station"
}

reversed_action_mapping = {
    "search for hotel": "find_hotel",
    "book hotel": "book_hotel",
    "search for trains": "find_train",
    "book train ticket": "book_train",
    "search for attractions": "find_attraction",
    "search for restaurants": "find_restaurant",
    "book table at restaurant": "book_restaurant",
    "search for hospital": "find_hospital",
    "book taxi": "book_taxi",
    "search for taxi": "find_taxi",
    "search for bus": "find_bus",
    "search for police station": "find_police"
}


def call_LLM_gpt3_abcd(dialogue, Action):
    abcd_hint_prompt = """
    The following are conversations between a user and an assistant. Indicated by the dialog acts, the assistant can help the user with checking in or providing information of temperature, time, price, location, and so on.
    The response should be coherent, engaging, diverse, informative, and overall good and should be in line with the next action.
    The response should be concise and to the point and not exceed 30 words. If there is a slot value, such as <item>, <username>, it should be filled in with the correct value.

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
    Assistant(pull-up-account [albert sanders]): account has been pulled up for albert sanders.

    Conversation:
    Context: Context: hello, how may i help you? i want to know the state of my refund. let me help you with that. i have an existing refund of $100 + i want to refund another $<amount>. did you want to add an extra item to your current refund? yes. could i have your full name or account id? albert sanders. account id 123445. thanks. could i have your username, email address and order id to validate your order? <username>. <email>. and the order id? <order_id>. thank you. what is the  item that you want to return? jeans. <name>. 
    Assistant(record-reason [guess jeans]): a reason of guess jeans has been recorded.

    Conversation:
    Context: hi. i want to manage my shipping details as my situation has changed. welcome to acmebrands! how may i help you today? i see. what is your name please? i want to change my shipping address. rodriguez domingo. and what is the shipping status please? order received. thanks.
    Assistant(shipping-status [order received]): shipping status of order received has been noted.

    Conversation:
    Context: i would like to know more about a product. hello. how may i help you today? sure. i would like to know if the buttons are brown or black. i see. so you are looking to purchase buttons? is there a drop down menu to select the color buttons you want to buy? no im looking to buy a shirt and asking if the button on the shirt is brown or black. product: shirt  brand: michael_kors  amount: $<amount>. oh the buttons on a shirt? should have mentioned that at the beginning.  let me take a look for you. that shirt has dark brown buttons on them. 
    Assistant(select-faq [shirt_other_3]): faq answer related to shirt_other_3 was selected.

    Conversation: 
    [[DIALOG]]
    """
    abcd_hint_prompt = abcd_hint_prompt.strip()

    prompt = abcd_hint_prompt.replace("[[DIALOG]]", dialogue)

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant. You can generate a response to the user's input based on the given previous dialogue and the next action."})
    messages.append({"role": "user", "content": prompt})

    response = clientGPT.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response.choices[0].message.content)

    return response.choices[0].message.content

def call_LLM_gpt3_multiwoz(dialogue, Action):
    abcd_hint_prompt = """
    The following are conversations between a user and an assistant. Indicated by the dialog acts, the assistant can help the user with checking in or providing information of temperature, time, price, location, and so on.
    The response should be coherent, engaging, diverse, informative, and overall good and should be in line with the next action.
    The response should be concise and to the point and not exceed 30 words. If there is a slot, such as <item>, <username>, <location>, it should be filled in with the correct value.

    Available Dialog acts:
    - search for hotel: customers are looking for <price> hotels with <requirements>, <level>, in <location>, <date> <time>, the hotel should have <requirements>.
    - book hotel: customers are going to booking hotels for <number> people, <number> nights starting from <date>.
    - search for trains: customers are looking for trains from <location> to <location> on <date> <time>.
    - book train ticket: customers are going to booking train tickets for <number> people.
    - search for attractions: customers are looking for <type> attractions in <location> with <requirements>.
    - search for restaurants: customers are looking for <type> restaurants in <location>, <price> range, with <requirements>.
    - book table at restaurant: customers are going to booking tables at restaurants for <number> people, on <date> at <time>.
    - search for hospital: customers are looking for <type> hospitals in <location>.
    - book taxi: customers are going to booking taxis
    - search for taxi: customers are looking for a taxi at <time> from <location> to <location>.
    - search for bus: customers are looking for a bus from <location> to <location> on <date> <time>.
    - search for police station: customers are looking for police stations

    Conversation: 
    Context: i need a list of cheap place -s to stay that include free parking . alexander bed and breakfast is in the cheap price range in the centre of town . okay , does that place include free wifi and it is 4 stars ? yes , the alexander has free wifi and is a 4 star hotel . how many nights will you be staying ? i will be staying 5 nights starting from saturday .
    Assistant(search for hotel [with parking, cheap, with internet, alexander bed and breakfast, 4 stars]): customers are looking for cheap hotels with free parking and wifi, 4 stars, in the centre of town, for 5 nights starting from saturday

    Conversation:
    Context: i need a list of cheap place -s to stay that include free parking . alexander bed and breakfast is in the cheap price range in the centre of town . okay , does that place include free wifi and it is 4 stars ? yes , the alexander has free wifi and is a 4 star hotel . how many nights will you be staying ? i will be staying 5 nights starting from saturday . thank you so much for that information . how many people in your party so i can make the booking for you ? yes , can you book it for 6 people ?
    Assistant(book hotel [saturday, 6 people, 5 stay]): customers are going to booking hotels for 6 people, 5 nights starting from saturday.

    Conversation:
    Context: i need a list of cheap place -s to stay that include free parking . alexander bed and breakfast is in the cheap price range in the centre of town . okay , does that place include free wifi and it is 4 stars ? yes , the alexander has free wifi and is a 4 star hotel . how many nights will you be staying ? i will be staying 5 nights starting from saturday . thank you so much for that information . how many people in your party so i can make the booking for you ? yes , can you book it for 6 people ? booking was successful . your reference number is qit1w09h . do you need anything else ? thank you . i also need a train to kings lynn that leaves on sunday . there are 19 trains leaving cambridge on sunday . when would you like to depart ? i want the train to leave after 16:30 .
    Assistant(search for trains [sunday, kings lynn, 16:30]): customers are looking for trains to kings lynn on sunday, leaving after 16:30

    Conversation:
    Context: what trains arrive in cambridge by 10:30 ? where are you traveling from and on what day ? i am traveling from kings lynn on sunday . please give me your day and time of departure to help me to narrow down to a suitable result . i do not have a departure time . i just need to be in cambridge by 10:30 . i suggest the tr8092 that will arrive in cambridge at 08:58 . this will give you ample time . do you want me to book this ? please book for 5 people , i will also need the reference number .
    Assistant(book train ticket [5 people]): customers are going to booking train tickets for 5 people.
    
    Conversation:
    Context: hello , i am looking for something to do in the west part of town . it sould involve multiple sports . unfortunately none of those place -s exist here . any other preferences ? hm , can you tell me about what entertainment venue -s might be on the west side of town instead ? there s a fun place called whale of a time at unit 8 , viking way , bar hill . thank you , can i please get an address and postal code . is there an entrance fee that will be charged .
    Assistant(search for attractions [west, entertainment, whale of a time]): customers are looking for entertainment venues on the west side of town, whale of a time

    Conversation:
    Context: i am looking for a chinese restaurant please . or 1 that serves chinese food . what area of town would you like the restaurant to be in ? i would like to be in the centre of town . jinling noodle bar is in the centre area serving chinese in the moderate price range . would you like to reserve a table ? i want something expensive .
    Assistant(search for restaurants [chinese, centre, expensive]): customers are looking for chinese restaurants in the centre of town, expensive

    Conversation:
    Context: can you let me know if a place called the gonville hotel is still around ? yes and it is as popular as ever . it is 3 stars and quite expensive . would you like me to book a room for you ? what area of town is it in ? it s in town centre . ok , thanks . also , are there any indian restaurant -s in the centre ? there are several with various price range -s . whatever you recommend . give me your best recommendation and go ahead and book me for a table for 6 people . i want to go on saturday at 15:30 . saffron brasserie has your reservation . it will be held for 15 minutes . the reference number is , pryp175n . can i get you the address or phone number ? no , that will be all . thank you !
    Assistant(book table at restaurant [saturday, 6 people, 15:30]): customers are going to booking tables at restaurants for 6 people, on saturday at 15:30

    Conversation:
    Context: i need to find a hospital here in the area . the nearest hospital is located at hills rd , cambridge . can i get thhe phone number and postcode which department are you looking for so i can give you the correct phone number ? i need the paediatric clinic please . the phone number for the paediatric clinic is 01223348313 , is there anything else i can help with ? yes , may i please have the post code ?
    Assistant(search for hospital [paediatric clinic]): customers are looking for the paediatric clinic.

    Conversation:
    Context: i am looking for a hotel in cambridge called the cambridge belfry that hotel is in the west and listed as cheap , but still has 4 stars . they provide free parking and internet . would you like to make a reservation ? yes , could you please book me a room on tuesday for 5 people and for 4 nights . booking was successful . reference number: mcitlhi8 . great ! i also need information on multiple sports in the centre . there are no multiple sports attractions in the centre . should we try another area ? how about a museum ? how about broughton house gallery ? it s free of charge . that sounds great . can you please give me the phone number ? sure . their phone number is 01223314960 . i would also like a taxi to commute and i would like it 17:45 please and the contact number and the car type , thank you okay , where do you want the taxi to pick you up and where will you be going ? i need the taxi to pick me up at the hotel by 17:45 .
    Assistant(search for taxi [17:45, cambridge belfry, broughton house gallery]): customers are looking for a taxi at 17:45 from cambridge belfry to broughton house gallery.

    Conversation:
    Context: can you help me find a train that leaves cambridge after 9:45 pm ? thanks . i can help with that . what is the destination and what day would you like to travel ? i am going to bishops storford on wednesday . i actually need to leave after 21:45 though . i do not have any trains that match your request . that s disappointing . can you recommend a taxi or bus service ?
    Assistant(search for bus [wednesday, cambridge, bishops stortford, 21:45]): customers are looking for a bus from cambridge to bishops stortford on wednesday, leaving after 21:45.

    Conversation:
    Context: i am looking for the parkside police station parkside police station is located in parkside , cambridge , within the postcode of cb11jg . may i help with something else ? yes , can you please provide their phone number and physical address ?
    Assistant(search for police station [none]): customers are looking for police stations.

    Conversation: 
    [[DIALOG]]
    """
    abcd_hint_prompt = abcd_hint_prompt.strip()

    prompt = abcd_hint_prompt.replace("[[DIALOG]]", dialogue)

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant. You can generate a response to the user's input based on the given previous dialogue and the next action."})
    messages.append({"role": "user", "content": prompt})

    response = clientGPT.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response.choices[0].message.content)

    return response.choices[0].message.content

def call_LLM_gpt4(dialogue, Action):
    abcd_hint_prompt = """
    The following are conversations between a user and an assistant. Indicated by the dialog acts, the assistant can help the user with checking in or providing information of temperature, time, price, location, and so on.
    The response should be coherent, engaging, diverse, informative, and overall good and should be in line with the next action.
    The response should be concise and to the point and not exceed 30 words. If there is a slot value, such as <item>, <username>, it should be filled in with the correct value.

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
    Assistant(pull-up-account [albert sanders]): account has been pulled up for albert sanders.

    Conversation:
    Context: Context: hello, how may i help you? i want to know the state of my refund. let me help you with that. i have an existing refund of $100 + i want to refund another $<amount>. did you want to add an extra item to your current refund? yes. could i have your full name or account id? albert sanders. account id 123445. thanks. could i have your username, email address and order id to validate your order? <username>. <email>. and the order id? <order_id>. thank you. what is the  item that you want to return? jeans. <name>. 
    Assistant(record-reason [guess jeans]): a reason of guess jeans has been recorded.

    Conversation:
    Context: hi. i want to manage my shipping details as my situation has changed. welcome to acmebrands! how may i help you today? i see. what is your name please? i want to change my shipping address. rodriguez domingo. and what is the shipping status please? order received. thanks.
    Assistant(shipping-status [order received]): shipping status of order received has been noted.

    Conversation:
    Context: i would like to know more about a product. hello. how may i help you today? sure. i would like to know if the buttons are brown or black. i see. so you are looking to purchase buttons? is there a drop down menu to select the color buttons you want to buy? no im looking to buy a shirt and asking if the button on the shirt is brown or black. product: shirt  brand: michael_kors  amount: $<amount>. oh the buttons on a shirt? should have mentioned that at the beginning.  let me take a look for you. that shirt has dark brown buttons on them. 
    Assistant(select-faq [shirt_other_3]): faq answer related to shirt_other_3 was selected.

    Conversation: 
    [[DIALOG]]
    """
    abcd_hint_prompt = abcd_hint_prompt.strip()

    prompt = abcd_hint_prompt.replace("[[DIALOG]]", dialogue)

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant. You can generate a response to the user's input based on the given previous dialogue and the next action."})
    messages.append({"role": "user", "content": prompt})

    response = clientGPT.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response.choices[0].message.content)

    return response.choices[0].message.content

def call_LLM_gpt4turbo(dialogue, Action):
    abcd_hint_prompt = """
    The following are conversations between a user and an assistant. Indicated by the dialog acts, the assistant can help the user with checking in or providing information of temperature, time, price, location, and so on.
    The response should be coherent, engaging, diverse, informative, and overall good and should be in line with the next action.
    The response should be concise and to the point and not exceed 30 words. If there is a slot value, such as <item>, <username>, it should be filled in with the correct value.

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
    Assistant(pull-up-account [albert sanders]): account has been pulled up for albert sanders.

    Conversation:
    Context: Context: hello, how may i help you? i want to know the state of my refund. let me help you with that. i have an existing refund of $100 + i want to refund another $<amount>. did you want to add an extra item to your current refund? yes. could i have your full name or account id? albert sanders. account id 123445. thanks. could i have your username, email address and order id to validate your order? <username>. <email>. and the order id? <order_id>. thank you. what is the  item that you want to return? jeans. <name>. 
    Assistant(record-reason [guess jeans]): a reason of guess jeans has been recorded.

    Conversation:
    Context: hi. i want to manage my shipping details as my situation has changed. welcome to acmebrands! how may i help you today? i see. what is your name please? i want to change my shipping address. rodriguez domingo. and what is the shipping status please? order received. thanks.
    Assistant(shipping-status [order received]): shipping status of order received has been noted.

    Conversation:
    Context: i would like to know more about a product. hello. how may i help you today? sure. i would like to know if the buttons are brown or black. i see. so you are looking to purchase buttons? is there a drop down menu to select the color buttons you want to buy? no im looking to buy a shirt and asking if the button on the shirt is brown or black. product: shirt  brand: michael_kors  amount: $<amount>. oh the buttons on a shirt? should have mentioned that at the beginning.  let me take a look for you. that shirt has dark brown buttons on them. 
    Assistant(select-faq [shirt_other_3]): faq answer related to shirt_other_3 was selected.

    Conversation: 
    [[DIALOG]]
    """
    abcd_hint_prompt = abcd_hint_prompt.strip()

    prompt = abcd_hint_prompt.replace("[[DIALOG]]", dialogue)

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant. You can generate a response to the user's input based on the given previous dialogue and the next action."})
    messages.append({"role": "user", "content": prompt})

    response = clientGPT.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response.choices[0].message.content)

    return response.choices[0].message.content

def gen_dial(args, context, pred_action_value, label_action_value, convo_id, turn_id, counter, save_file_path):
    dialog_with_hint = ""
    # for each in context_list:
    #     dialog_with_hint += "User: " + each["user"] + "\n" + "Assistant(" + each["action"] + "): " + each["agent"] + "\n"
    # dialog_with_hint += "User: " + context + "\n" + "Assistant(" + action + "): "
    dialog_with_hint += f"{context}\nAssistant({pred_action_value}):"

    for try_time in range(3):
        try:
            if args.model == "gpt35":
                if args.dataset == "abcd":
                    response = call_LLM_gpt3_abcd(dialog_with_hint, pred_action_value)
                elif args.dataset == "multiwoz":
                    response = call_LLM_gpt3_multiwoz(dialog_with_hint, pred_action_value)
            elif args.model == "gpt4":
                response = call_LLM_gpt4(dialog_with_hint, pred_action_value)
            elif args.model == "gpt4turbo":
                response = call_LLM_gpt4turbo(dialog_with_hint, pred_action_value)
            break
        except:
            if try_time == 2:
                response = "MISSING"
                print("cannot get response from LLM")
            print("retrying ...")
            time.sleep(1)
            continue

    save_context = {"sample_id": counter, "convo_id": convo_id, "turn_id": turn_id, "target": label_action_value, "input": context, "pred_action": pred_action_value, "pred_utterance": "MISSING"}
    save_context["pred_utterance"] = response
    
    action_name = label_action_value.split(" [")[0].strip()
    values = label_action_value.split(" [")[1].replace("]", "").strip()
    if action_name in reversed_action_mapping:
        action_name = reversed_action_mapping[action_name]
        save_context["label_utterance"] = action_description[action_name] + " " + values + "."
    else:
        save_context["label_utterance"] = "MISSING"

    with open(save_file_path, "a") as w:
        json.dump(save_context, w)
        w.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="abcd")
    parser.add_argument('--model', type=str, default="gpt35", help="gpt-3.5-turbo or gpt-4 or gpt-4-turbo")
    parser.add_argument('--waction', action="store_true", help="with or without actions")
    parser.add_argument('--test_file', type=str, default="/research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/data/processed/test_AST_abcd_woaction_flow_all.json")
    parser.add_argument('--file_suffix', type=str, default="Trial1 or Trial2 or Trial3")

    args = parser.parse_args()

    save_file_path = f"dialogues/{args.dataset}_waction_{args.model}_{args.file_suffix}.json" if args.waction else f"dialogues/{args.dataset}_woaction_{args.model}_{args.file_suffix}.json"

    if os.path.exists(save_file_path):
        os.remove(save_file_path)

    convo_ids = []
    turn_counts = []
    contexts = []
    labels = []
    flows = []
    predictions = []
    sample_ids = []
    response_file = args.test_file
    with jsonlines.open(response_file) as reader:
        for sample in reader:
            sample_ids.append(sample["sample_id"])
            convo_ids.append(sample["convo_id"])
            turn_counts.append(sample["turn_id"])
            contexts.append(sample["input"])
            labels.append(sample["target"])
            predictions.append(sample["pred_action"])

    for i in tqdm(range(len(predictions))):
        pred_action = predictions[i]
        label_action = labels[i]
        context = contexts[i]
        convo_id = convo_ids[i]
        turn_id = turn_counts[i]
        sample_id = sample_ids[i]

        gen_dial(args, context, pred_action, label_action, convo_id, turn_id, sample_id, save_file_path)