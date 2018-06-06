
nums = [4, 14, 2]
for b in map('{:032b}'.format, nums):
    print(b)
for b in zip(*map('{:032b}'.format, nums)):
    print(b)

for b in zip([1,2,3,10],[4,5,6,12],[7,8,9]):
    print(b)