#!/usr/bin/env python3
"""
Interactive prime number generator.

This program allows users to specify how many prime numbers they want to compute
and provides various display options.
"""

import sys
import time
from prime_numbers import get_first_n_primes, is_prime


def display_primes(primes, columns=10):
    """
    Display primes in a formatted table.
    
    Args:
        primes (list): List of prime numbers
        columns (int): Number of columns for display
    """
    print(f"\nThe first {len(primes)} prime numbers are:")
    print("=" * (columns * 6))
    
    for i in range(0, len(primes), columns):
        row = primes[i:i+columns]
        print(" ".join(f"{prime:5d}" for prime in row))
    
    print("=" * (columns * 6))


def show_statistics(primes):
    """
    Show statistics about the prime numbers.
    
    Args:
        primes (list): List of prime numbers
    """
    if not primes:
        return
    
    print(f"\nStatistics:")
    print(f"Count: {len(primes)}")
    print(f"Smallest: {primes[0]}")
    print(f"Largest: {primes[-1]}")
    print(f"Sum: {sum(primes)}")
    print(f"Average: {sum(primes) / len(primes):.2f}")
    
    # Find gaps between consecutive primes
    if len(primes) > 1:
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        print(f"Largest gap: {max(gaps)}")
        print(f"Average gap: {sum(gaps) / len(gaps):.2f}")


def prime_checker():
    """Interactive prime checker."""
    print("\n--- Prime Checker ---")
    while True:
        try:
            user_input = input("Enter a number to check if it's prime (or 'q' to quit): ").strip()
            if user_input.lower() == 'q':
                break
            
            num = int(user_input)
            if num < 0:
                print("Please enter a non-negative number.")
                continue
            
            if is_prime(num):
                print(f"{num} is prime! ✓")
            else:
                print(f"{num} is not prime. ✗")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")
        except KeyboardInterrupt:
            break


def main():
    """Main interactive function."""
    print("🔢 Prime Number Generator")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Generate first N prime numbers")
        print("2. Check if a number is prime")
        print("3. Exit")
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                try:
                    n = int(input("How many prime numbers do you want? "))
                    if n <= 0:
                        print("Please enter a positive number.")
                        continue
                    
                    print(f"\nComputing the first {n} prime numbers...")
                    start_time = time.time()
                    
                    primes = get_first_n_primes(n)
                    
                    end_time = time.time()
                    
                    # Choose appropriate number of columns based on n
                    columns = min(10, max(5, n // 10 + 1))
                    display_primes(primes, columns)
                    show_statistics(primes)
                    
                    print(f"\nComputation time: {end_time - start_time:.4f} seconds")
                    
                except ValueError:
                    print("Please enter a valid number.")
                    
            elif choice == '2':
                prime_checker()
                
            elif choice == '3':
                print("Goodbye! 👋")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except EOFError:
            print("\n\nGoodbye! 👋")
            break


if __name__ == "__main__":
    main()