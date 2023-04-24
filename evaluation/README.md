# Question 1: <br>
Assign the inp variable to the input number. The program prints prime and composite if the number is not divisible ot divisible for any number from 2 to inp-1. 
Note: We do not have break and continue statements in lirix and thus, break could not be added to stop the loop as we find one divisor of the number and the loop runs till the end. 

Result :- For a number to be composite, there should be atleast one composite printed in the output console. If all the print statements print prime, the number is prime. 

# Question 2:  <br>
Pass the string to be evaluated in the inp variable. 

Result :- Number of vowels will directly be printed in the console.

# Question 3: <br>
Enter the first number in variable a and second number in variable b. 

Result :- The gcd of both numbers will directly be printed. 

# Question 4: <br>
Pass the input sentence in variable inp. 

Results: The number of words will be printed as the output. 

# Question 6: <br>
Enter the input string in the variable 's'. 

Result :- The output will be True if s is a palindrome and False if it is not. 

# Question 7: <br>
Enter the length of the list in the first line. 
array arr 5; -> this line initialises an array of length 5. Pass the length of required input array instead of 5.
When this array is initialised, all its elements are initialised to 0. In order to make the desired array, use update to make the elements of the array as desired. 
To make the array [1,2,3,4,5], we have used the following statements:
update arr 1 1;
update arr 2 2;
update arr 3 3;
update arr 4 4;
update arr 5 5;

This will create an array : [1, 2, 3, 4, 5].

To enter the target number, use k. Pass the target number in the variable k.

Also change the ending limits of the nested loops. 
Presently, since the length of the array is 5, the loops are: 
for i from 1 to 5{
	for j from 1 to 5{
        (body)
    };
};

Change 5 to the length of the input array.

Result :- The numbers in the array that will sum to the target variable will be printed one after the other in the console. 
In case of multiple answers, all such pairs will be printed in the console onr after the other. 
For Eg: array = [1, 2, 3, 4, 5] and k = 9
Output:
4
5
5
4
There are two pairs - (4, 5) and (5, 4) that will be printed as output that sum up to the target variable (==9 here). .

# Question 8: <br>
Similar to the above question, initialise the array,
array arr 5;
update arr 1 1;
update arr 2 2;
update arr 3 3;
update arr 4 4;
update arr 5 5;

We have written the above code to form an array : [1, 2, 3, 4, 5]
Also change the ending limit of the loop. Since the length of the array we have tested is 5, we have defined the loop as:
for i from 1 to 5{
    (body)
};

Change 5 to the length of the input array. 

Result :- The sum of elements of numbers will directly be printed in the console. 

# Question 13: <br>
array arr 5; -> this line initialises an array of length 5. Pass the length of required input array instead of 5.
When this array is initialised, all its elements are initialised to 0. In order to make the desired array, use update to make the elements of the array as desired. 
To make the array [2,7,9,3,5], we have used the following statements:
update arr 1 2;
update arr 2 7;
update arr 3 9;
update arr 4 3;
update arr 5 5;

This will create an array : [2,7,9,3,5].

Used a for loop to iterate over the array to find the largest number in the array. then store the largest value and again used a loop to iterate over the array and finding the second largest number using the condition in if statement.

Result :- The second largest number in the array will be printed out as output.

# Question 15: <br>
Assigning inp variable the input number. Using a loop to iterate from 1 to the input number to get the factorial.

Result :- The factorial of the input number

# Question 17: <br>
Assigning inp variable the input string. Calculated length of the input string and iterating over the range of 1 to length of string using for loop. stored each letter starting from the end to start and cocatenated on each iteration.

Result :- The string in a reverse order would be displayed as an output.

