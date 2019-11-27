import math
def consinants(line):
    con=0
    num=0
    for c in line:
        if c not in ['A','a','E','e','I','i','O','o','U','u']:
            con+=1
        num+=1
    return con/num
def avgLengthGreater5(line):
    words=line.split()
    sum=0
    for word in words:
        sum+=len(word)
    return sum/len(words)>5
def avgLengthLess5(line):
    words=line.split()
    sum=0
    for word in words:
        sum+=len(word)
    return sum/len(words)<5
def containsQ(line):
    for c in line:
        if c=='v' or c=='V':
            return True
    return False

def containsDE(line):
    words=line.split()
    return not "de" not in words