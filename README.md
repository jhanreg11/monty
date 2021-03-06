## MONTY-AI
Monty is a LSTM model created to generate Python3 code. It was inspired by Andrej Karpathy's blog post, 
[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
 
 
## Current Plan
1. Source data.
    - To get Monty to work well, I will need a ton of python files.
    - I plan to scrape them from large Github projects, obtained from [here](https://hackernoon.com/50-popular-python-open-source-projects-on-github-in-2018-c750f9bf56a0).
    - The main problem with this method will be that the model will see a lot of imports from potentially unavailable
    packages, which could cause bad code to be generated by Monty when he is trained.
    - Goal training data size: 20 MB.
2. Preprocess sourced data.
    - To give Monty the best chance to generate interpretable code, I will need to clean the raw python files thoroughly.
    - I will mainly remove unnecessary newlines, convert indents to single '\t' tokens, remove comments, add a fake end
    of file token, and remove import statements to unavailable packages.
    - Varying file length may cause a problem for training. If so, I will develop a method to safely truncate long files
    and ignore short files.
3. Build model & train.
    - The main purpose of this project is to practice my LSTM and Keras knowledge.
    - I plan to create a multi-layer LSTM model.
    - I will pay for server space in GCP to train the model quickly.
4. Build demo.
    - I want to build a cool interactive web demo where people can use Monty to sample random code and see if it runs.
    - I will build some sort of feature where users can run Monty's sampled code in the browser, view its output, and edit it.
    - I will also showcase some runnable examples for users if they have a hard time generating their own good code.
    
## Future improvements
- Create variations of Monty to generate different types of code (e.g. DL Models, games, GUI's)
- Use reinforcement learning (Deep Q, NEAT, etc) to create a model that answers coding questions. Rewards would be based
on the # of test cases passed, and punishments would depend on errors generated.