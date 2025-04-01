import sys
from src.regressionTree import RegressionTree
from src.regressionLinear import RegressionLinear

def main():
    print("Choose a regression model to run:")
    print("1. Regression Guillaume")
    print("2. Regression Tree")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        regression = RegressionLinear()
        regression.run()
    elif choice == '2':
        regression = RegressionTree()
        regression.run()
    else:
        print("Invalid choice. Please enter 1 or 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()