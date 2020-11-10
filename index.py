#!/usr/bin/python3
import os
import aiml

BRAIN_FILE="brain.txt"

k = aiml.Kernel()

# To increase the startup speed of the bot it is
# possible to save the parsed aiml files as a
# dump. This code checks if a dump exists and
# otherwise loads the aiml from the xml files
# and saves the brain dump.
if os.path.exists(BRAIN_FILE):
    print("Loading logged file from brain: " + BRAIN_FILE)
    k.loadBrain(BRAIN_FILE)
else:
    print("Parsing aiml files")
    k.bootstrap(learnFiles="fitness.aiml", commands="") #Loads the fitness.aiml file
    print("Saving brain file: " + BRAIN_FILE)
    k.saveBrain(BRAIN_FILE)

# Endless loop which passes the input to the bot and prints
# its response
while True:
    input_text = input(" Hey Guest User Send me a Message > ")
    response = k.respond(input_text)
    print(" Jurini Fitness Bot  >  " + response)

    if input_text == 'quit': 
       print ("Thank you for visiting.")
       break    