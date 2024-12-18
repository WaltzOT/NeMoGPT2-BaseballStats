import os
import sys

def prompt_menu():
    print("Welcome to the Baseball Chatbot Docker Service!")
    print("Select an option:")
    print("1. Train the chatbot")
    print("2. Test the chatbot")
    print("3. Exit")
    
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        print("Starting training...")
        os.system("python /workspace/src/initialTraining.py")
    elif choice == "2":
        print("Starting testing...")
        os.system("python /workspace/src/testingBot.py")
    elif choice == "3":
        print("Shutting down...")
        sys.exit(0)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        prompt_menu()

if __name__ == "__main__":
    prompt_menu()
