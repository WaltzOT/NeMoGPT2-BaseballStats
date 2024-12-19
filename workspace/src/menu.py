import os
import sys

def prompt_menu():
    while True:  # Keep showing the menu until the user exits
        print("\nWelcome to the Baseball Chatbot Docker Service!")
        print("Select an option:")
        print("1. Train a chatbot")
        print("2. Test a chatbot")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            bot_name = input("Enter a name for the bot you want to train: ").strip()
            print(f"Starting training for bot: {bot_name}...\n")
            os.system(f"python3 /workspace/src/initialTraining.py {bot_name}")
            print("\nTraining complete! Returning to menu...")
        elif choice == "2":
            bot_name = input("Enter the name of the bot to test: ").strip()
            print(f"Starting testing for bot: {bot_name}...\n")
            os.system(f"python3 /workspace/src/testingBot.py {bot_name}")
            print("\nTesting complete! Returning to menu...")
        elif choice == "3":
            print("Shutting down...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    prompt_menu()
