import os
import sys

def prompt_menu():
    while True:  # Keep showing the menu until the user exits
        print("\nWelcome to the Baseball Chatbot Docker Service!")
        print("Select an option:")
        print("1. Train the chatbot")
        print("2. Test the chatbot")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            print("Starting training...\n")
            os.system("python3 /workspace/src/initialTraining.py")
            print("\nTraining complete! Returning to menu...")
        elif choice == "2":
            print("Starting testing...\n")
            os.system("python3 /workspace/src/testingBot.py")
            print("\nTesting complete! Returning to menu...")
        elif choice == "3":
            print("Shutting down...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    prompt_menu()
