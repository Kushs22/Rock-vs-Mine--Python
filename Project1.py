#Number guessing game:
import random

def guess_number(): 
    number_to_guess = random.randint(1, 100) 
     # Generates a random number between 1 and 100 with 100 included using random.randrange(1,100) won't include the upper bound number 100.
    attempts = 0 #Initially the attempts of the user are set at 0
    
    print("Welcome to the Number Guessing Game!")
    print("I have chosen a number between 1 and 100. Try to guess it!")

    while True:
        try:
            user_guess = int(input("\nEnter your guess: "))
            attempts += 1 #As user has guessed a number so attempts increases by 1

            if user_guess < number_to_guess:
                print("Too low! Try a higher number.")
            elif user_guess > number_to_guess:
                print("Too high! Try a lower number.")
            else:
                print(f"Congratulations! You guessed the number {number_to_guess} in {attempts} attempts!")
                break

        except ValueError:
            print("Invalid input. Please enter a valid number.")

    
