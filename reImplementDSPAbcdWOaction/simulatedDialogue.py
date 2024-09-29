'''
TODO: Implement the simulated dialogue system
1. Automation: The system should be able to generate responses to user queries
Q: how to generate responses?
A: LLM as the response generator, and use the generated action as hints to generate the response
Q: how to automate the queries from users?
A: 1. Use the ground truth questions no matter how the LLM responds
    2. Manually generate the queries, but it is not really automated, and cannot estimate all the data.
    3. Use the LLM to generate the queries, but how to make sure the queries in line with the conversation objectiove?
'''