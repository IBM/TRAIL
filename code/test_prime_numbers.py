#!/usr/bin/env python3
"""
Test script for the prime numbers program.
"""

from prime_numbers import is_prime, get_first_n_primes, sieve_of_eratosthenes


def test_is_prime():
    """Test the is_prime function."""
    print("Testing is_prime function...")
    
    # Test known primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in primes:
        assert is_prime(p), f"{p} should be prime"
    
    # Test known non-primes
    non_primes = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22]
    for n in non_primes:
        assert not is_prime(n), f"{n} should not be prime"
    
    print("✓ is_prime function tests passed")


def test_sieve_of_eratosthenes():
    """Test the sieve_of_eratosthenes function."""
    print("Testing sieve_of_eratosthenes function...")
    
    # Test small cases
    assert sieve_of_eratosthenes(1) == []
    assert sieve_of_eratosthenes(2) == [2]
    assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]
    assert sieve_of_eratosthenes(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    print("✓ sieve_of_eratosthenes function tests passed")


def test_first_n_primes():
    """Test the get_first_n_primes function."""
    print("Testing get_first_n_primes function...")
    
    # Test small cases
    assert get_first_n_primes(0) == []
    assert get_first_n_primes(1) == [2]
    assert get_first_n_primes(5) == [2, 3, 5, 7, 11]
    assert get_first_n_primes(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    # Test that we get exactly 100 primes
    first_100 = get_first_n_primes(100)
    assert len(first_100) == 100
    assert first_100[0] == 2  # First prime
    assert first_100[-1] == 541  # 100th prime
    
    # Verify all returned numbers are actually prime
    for p in first_100:
        assert is_prime(p), f"{p} should be prime"
    
    print("✓ get_first_n_primes function tests passed")


def verify_known_results():
    """Verify against known mathematical results."""
    print("Verifying against known mathematical results...")
    
    first_100 = get_first_n_primes(100)
    
    # Known facts about the first 100 primes
    assert first_100[0] == 2, "First prime should be 2"
    assert first_100[9] == 29, "10th prime should be 29"
    assert first_100[24] == 97, "25th prime should be 97"
    assert first_100[49] == 229, "50th prime should be 229"
    assert first_100[99] == 541, "100th prime should be 541"
    
    # Sum of first 100 primes is known to be 24133
    assert sum(first_100) == 24133, "Sum of first 100 primes should be 24133"
    
    print("✓ Known mathematical results verified")


def main():
    """Run all tests."""
    print("Running tests for prime numbers program...\n")
    
    test_is_prime()
    test_sieve_of_eratosthenes()
    test_first_n_primes()
    verify_known_results()
    
    print("\n🎉 All tests passed! The prime numbers program is working correctly.")


if __name__ == "__main__":
    main()