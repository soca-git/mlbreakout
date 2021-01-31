
#############
### SETUP ###
#############

git clone ---> https://github.com/Ankita03060/Machine-learning-.git

pip install -r requirements.txt

pip install gym[atari]
pip install git+https://github.com/Kojoley/atari-py.git (if windows)

python breakout.py --mode train/test

#############
### NOTES ###
#############

--- analytics

Contains some spreadsheets with data from previous training run.

--- trained data

Final weights, training log file and video clip of trained agent.

--- rl

Core classes and code for implementation.
- memory.py, callbacks.py, policy.py & util.py were slightly modified from original repo.
- models.py, processors.py are new
- agent.py & agentcore.py are new and implement a Q Deep-Learning agent
