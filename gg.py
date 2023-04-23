def collatz_sequence_length(n):
    """Return the length of the Collatz sequence starting with n."""
    length = 1
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        length += 1
    return length

max_length = 0
max_starting_number = 0

for starting_number in range(1, 1000000):
    length = collatz_sequence_length(starting_number)
    if length > max_length:
        max_length = length
        max_starting_number = starting_number

print(max_length)
print(max_starting_number)
