import itertools as it
import os
#os.path.abspath("C:/Users/Pubudu/Desktop/New folder")
os.chdir("C:/Users/Pubudu/Desktop/New folder")
f = open("data.txt", "w")

# Python 3 program to print all
# possible strings of length k

# The method that prints all
# possible strings of length k.
# It is mainly a wrapper over
# recursive function printAllKLengthRec()


def printAllKLength(set, k):

    n = len(set)
    printAllKLengthRec(set, "", n, k)


# The main recursive method
# to print all possible
# strings of length k
xxx = set()


def printAllKLengthRec(set, prefix, n, k):

    # Base case: k is 0,
    # print prefix
    if (k == 0):
        # print(prefix)
        xxx.add(prefix)
        return

    # One by one add all characters
    # from set and recursively
    # call for k equals to k-1
    for i in range(n):

        # Next character of input added
        newPrefix = prefix + set[i]

        # k is decreased, because
        # we have added a new character
        printAllKLengthRec(set, newPrefix, n, k - 1)


# Driver Code
if __name__ == "__main__":

    # set2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
    #         'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    set2 = ['a', 'b', 'c', 'd', 'e']

    k = 5
    printAllKLength(set2, k)

# This code is contributed
# by ChitraNayal

# print(len(xxx))
xxx = list(xxx)
yyy = ['A', 'B', 'C', 'D', 'E']

z = list(it.product(xxx, yyy))

for i in range(len(z)):
    # print(i[0]+i[1])
    x1 = str(str(z[i][0])+str(z[i][1])+" "+str(i))

    # f.write(i[0]+i[1])
    f.write(x1)
    f.write("\n")
print(len(z))
f.close()
