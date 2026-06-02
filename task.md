Task: On a single h100, get the lowest fineweb validation loss possible in 5 mins of training time. No changes to the data allowed. No looking at the future versions of the repo readme or other nanogpt speedrun records beyond the current commit (commit 7), but you can use everything else in the ML literature.

Keep an updated report indicating what you are doing and trying, and how its going so far along with plots. At the end of the two hours, a human should be able to understand what you did, and how well your final solution did on the task.

Time: 2.5 hours

Hardware: 4xH100 (so you can run up to 4 experiments in parallel)

Setup:
- pip install -r requirements.txt
- python data/cached_fineweb10B.py 27 # downloads only the first 2.7B training tokens to save time
