rows = 4
columns = 3
number = 1

for i in range(rows):
    for j in range(columns):
        print(number, end=" ")
        number += 1
    print()  # Move to the next line after each row
