# travelling salesman problem using genetic algorithm


from random import randrange

start=1
n=1
dis=[[]]


def population(size):
    assert size<=__import__('math').factorial(n)
    p=set()
    while len(p)<size:
        x=[start]
        b=[0]*n
        b[start]=1
        r=randrange(n)
        while len(x)<n:
            while b[r]: r=randrange(n)
            b[r]=1
            x.append(r)
        x.append(start)
        p.add(tuple(x))
    return [list(i) for i in p]

def score(X:list[int]):
    if len(set(X))!=n or not X[0]==X[-1]==start:
        return float('inf')
    return abs(sum(dis[X[i-1]][X[i]] for i in range(1,n+1)))

def fitness(population):
    return [*map(score,population)]

def crossover(x,y):
    # randomly add a subarray from y to x and fill remaining ones from x
    i,j=randrange(1,n),randrange(1,n)
    if i>j: i,j=j,i
    c=[0]*len(x)
    b=[False]*n
    for k in range(i,j+1):
        c[k]=y[k]
        b[y[k]]=True
    ix=0
    for z in ((0,i),(j+1,n)):
        for k in range(*z):
            while b[x[ix]]: ix+=1
            b[x[ix]]=True
            c[k]=x[ix]
    c[-1]=start
    return c

def mutate(x):
    i,j=randrange(1,n),randrange(1,n)
    x[i],x[j] = x[j],x[i]

def selection(population,fitness,_top=33):
    i=randrange(len(fitness)//3)
    return (sorted(zip(fitness,population))[i])[1] # top 33% default

def genetic_algo(popu,max_iter=10000):
    fit=fitness(popu)
    from tqdm import tqdm
    for _ in tqdm(range(max_iter)):
        next_gen=[]
        for j in range(len(popu)):
            x=selection(popu,fit)
            y=selection(popu,fit)
            child=crossover(x,y)
            if randrange(10101)%3==0:
                mutate(child)
            next_gen.append(child)
        popu=next_gen
        fit=fitness(popu)
    return sorted(popu,key=score)[0]


def main():
    random_graph(1)
    p=population(300)
    x=genetic_algo(p,1000)
    print(x)
    print(score(x))


def random_graph(show=False):
    global n,dis,start
    n=randrange(15,100)
    start=randrange(n)
    dis=[[0]*n for i in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            dis[i][j]=dis[j][i]=randrange(1,51)
    if show:
        print(n)
        print(start)
        for i in range(n):
            for j in range(i+1,n):
                print(i,j,dis[i][j])
                # print(f'{i} -> {j} : {dis[i][j]}')



main()
