# Prime Number Programs

This directory contains several Python programs for computing and working with prime numbers.

## Files

### 1. `prime_numbers.py`
The main program that computes the first 100 prime numbers using the Sieve of Eratosthenes algorithm.

**Features:**
- Efficient prime number generation using the Sieve of Eratosthenes
- Displays the first 100 primes in a formatted table
- Shows statistics including sum and average
- Includes helper functions for prime checking and generation

**Usage:**
```bash
python prime_numbers.py
```

**Output:**
```
Computing the first 100 prime numbers...

The first 100 prime numbers are:
==================================================
   2    3    5    7   11   13   17   19   23   29
  31   37   41   43   47   53   59   61   67   71
  ...
==================================================
Total count: 100 primes
Largest prime: 541

Statistics:
Sum of first 100 primes: 24133
Average: 241.33
```

### 2. `interactive_primes.py`
An interactive version that allows users to specify how many primes they want and includes a prime checker.

**Features:**
- Interactive menu system
- Generate any number of prime numbers
- Prime number checker
- Performance timing
- Detailed statistics including gap analysis

**Usage:**
```bash
python interactive_primes.py
```

### 3. `test_prime_numbers.py`
Comprehensive test suite for the prime number functions.

**Features:**
- Tests for all prime number functions
- Verification against known mathematical results
- Ensures correctness of the 100th prime (541)
- Validates the sum of first 100 primes (24133)

**Usage:**
```bash
python test_prime_numbers.py
```

## Algorithm Details

### Sieve of Eratosthenes
The main algorithm used is the Sieve of Eratosthenes, which is one of the most efficient ways to find all prime numbers up to a given limit.

**Time Complexity:** O(n log log n)
**Space Complexity:** O(n)

**How it works:**
1. Create a list of consecutive integers from 2 through n
2. Mark all multiples of each prime as composite
3. The remaining unmarked numbers are prime

### Adaptive Limit Finding
For finding the first N primes, the program uses an adaptive approach:
1. Estimates the upper bound using the approximation: nth prime ≈ n × ln(n)
2. If not enough primes are found, doubles the limit and tries again
3. Returns exactly the first N primes

## Mathematical Facts

- The 100th prime number is 541
- The sum of the first 100 prime numbers is 24,133
- The average of the first 100 primes is 241.33
- The largest gap between consecutive primes in the first 100 is 6 (between 887 and 907, though these are beyond the first 100)

## Performance

The programs are optimized for efficiency:
- Computing the first 100 primes typically takes less than 1ms
- Computing the first 1000 primes takes only a few milliseconds
- The sieve algorithm scales well for larger numbers

## Examples

### Finding the first 10 primes:
```python
from prime_numbers import get_first_n_primes
primes = get_first_n_primes(10)
print(primes)  # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

### Checking if a number is prime:
```python
from prime_numbers import is_prime
print(is_prime(17))  # True
print(is_prime(18))  # False
```

### Using the sieve directly:
```python
from prime_numbers import sieve_of_eratosthenes
primes_up_to_30 = sieve_of_eratosthenes(30)
print(primes_up_to_30)  # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```