#!/usr/bin/env python3
"""
Program to compute the first 100 prime numbers.

This program implements the Sieve of Eratosthenes algorithm to efficiently
find prime numbers, then extracts the first 100 primes.
"""

def sieve_of_eratosthenes(limit):
    """
    Generate all prime numbers up to a given limit using the Sieve of Eratosthenes.
    
    Args:
        limit (int): Upper bound for finding primes
        
    Returns:
        list: List of prime numbers up to the limit
    """
    if limit < 2:
        return []
    
    # Create a boolean array and initialize all entries as True
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False  # 0 and 1 are not prime
    
    p = 2
    while p * p <= limit:
        if is_prime[p]:
            # Mark all multiples of p as not prime
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
        p += 1
    
    # Collect all prime numbers
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    return primes


def is_prime(n):
    """
    Check if a number is prime using trial division.
    
    Args:
        n (int): Number to check
        
    Returns:
        bool: True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_first_n_primes(n):
    """
    Get the first n prime numbers.
    
    Args:
        n (int): Number of primes to find
        
    Returns:
        list: List of the first n prime numbers
    """
    if n <= 0:
        return []
    
    # Start with a reasonable estimate for the upper bound
    # The nth prime is approximately n * ln(n) for n >= 6
    if n < 6:
        limit = 15
    else:
        import math
        limit = int(n * math.log(n) * 1.5)  # Add some buffer
    
    primes = []
    while len(primes) < n:
        primes = sieve_of_eratosthenes(limit)
        if len(primes) < n:
            limit *= 2  # Double the limit and try again
    
    return primes[:n]


def main():
    """Main function to compute and display the first 100 prime numbers."""
    print("Computing the first 100 prime numbers...\n")
    
    # Get the first 100 prime numbers
    first_100_primes = get_first_n_primes(100)
    
    # Display the results
    print("The first 100 prime numbers are:")
    print("=" * 50)
    
    # Print in rows of 10 for better readability
    for i in range(0, len(first_100_primes), 10):
        row = first_100_primes[i:i+10]
        print(" ".join(f"{prime:4d}" for prime in row))
    
    print("=" * 50)
    print(f"Total count: {len(first_100_primes)} primes")
    print(f"Largest prime: {first_100_primes[-1]}")
    
    # Some interesting statistics
    print(f"\nStatistics:")
    print(f"Sum of first 100 primes: {sum(first_100_primes)}")
    print(f"Average: {sum(first_100_primes) / len(first_100_primes):.2f}")


if __name__ == "__main__":
    main()