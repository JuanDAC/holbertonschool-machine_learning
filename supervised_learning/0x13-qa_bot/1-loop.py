#!/usr/bin/env python3
"""
File that contains the function answer_loop
Create a script that takes in input from the user with the prompt 
Q: and prints A: as a response. If the user inputs exit, quit, goodbye,
or bye, case insensitive, print A: Goodbye and exit.
"""


if __name__ == "__main__":
    while (1):
        question = input("Q: ")
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            exit()
        print("A:")
