# NeMoGPT2-BaseballStats
Scripts used for training chatbots for baseball statistics in NeMo using GPT2. Running on CPU on and using Docker to complete tasks


- To run just type in bash "docker-compose build"
- Then "docker-compose run --service-ports nemogpt2-chatbot", in order to ensure the menu works. 
- Once done using just choose menu option 3, in which everything will shut down. 
- *WARNING* starting up a test or train is incredibly slow as it runs on CPU only, so give it time to run it's course, it looks like it's doing nothing at first but eventually will get there.


If you'd like to use your own data please just upload it into the DATA folder and keep the same naming format and add it in, but remove the old data while doing so.
