'''Write a program non-recursive and recursive program to calculate 
Fibonacci numbers and analyze their time and space complexity.'''

def recur(n):
   if n <= 1:
       return n
   else:
       return(recur(n-1) + recur(n-2))
def iterative(n):
    a = 0
    b = 1
    print(a)
    print(b)
    for i in range(2, n):
        print(a + b)
        a, b = b, a + b
if __name__ == "__main__":
    num = int(input("Enter the nth number for series: "))
    if num <= 0:
       print("Please enter a positive integer")
    else:
       print("Fibonacci sequence with recursion:")
       for i in range(num):
           print(recur(i))
    print("Fibonacci series with Iteration:")
    iterative(num)
def recur(n):
   if n <= 1:
       return n
   else:
       return(recur(n-1) + recur(n-2))
def iterative(n):
    a = 0
    b = 1
    print(a)
    print(b)
    for i in range(2, n):
        print(a + b)
        a, b = b, a + b
if __name__ == "__main__":
    num = int(input("Enter the nth number for series: "))
    if num <= 0:
       print("Please enter a positive integer")
    else:
       print("Fibonacci sequence with recursion:")
       for i in range(num):
           print(recur(i))
    print("Fibonacci series with Iteration:")
    iterative(num)

