import random

N = 10

reviewer = [chr(ord('a') + i) for i in range(N)]
suitor = [chr(ord('A') + i) for i in range(N)]

for i in range(100):
    f = open('data/data_'+str(i)+'.txt', 'w')
    
    for j in range(N):
        f.write(chr(ord('A') + j) +' : ')
        random.shuffle(reviewer)
        f.write(' '.join(reviewer)+'\n')

    for j in range(N):
        f.write(chr(ord('a') + j) +' : ')
        random.shuffle(suitor)
        f.write(' '.join(suitor)+'\n')

    f.close()
        