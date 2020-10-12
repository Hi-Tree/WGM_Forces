def factorial(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
    

def factorial2(n):
    r = 1
    while (n >= 1):
        r *= n
        n = n-2
    return r
   
