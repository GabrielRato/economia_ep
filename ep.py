import plotly
from plotly.graph_objs import Scatter, Layout, Marker
import plotly.graph_objs as go
import random


def oferta_demanda():

    print 'Aplicando randomizacao no cenario...'
    # construir parametrizacao random e printar ela descrita
    # varia de 0 a 10
    gostos = []
    for i in range(3):
        gostos.append(random.randint(0,10))
    gostos = sorted(gostos)
    expectativa = 6

    traces = []
    for b in [0.5, 8]:
        for gosto in gostos:
            delta = (gosto + expectativa)/20.0
            if b == 8:
                delta = -1*delta
            axis_x = []
            axis_y = []
            for x in range(10):
                axis_x.append(x)
                axis_y.append(b + float(delta*x))

            trace = go.Scatter(
                x=axis_x,
                y=axis_y
            )
            traces.append(trace)

    plotly.offline.plot({
        "data": [trace for trace in traces],
        "layout": Layout(title="CURVA OFERTA & DEMANDA" )
    })


    py.iplot(data, filename='basic-bar')

def elasticidade():

    elasticidade = ['simples', 'perfeita_inelastica', 'perfeita_elastica']
    for ele in elasticidade:
        axis_x = []
        traces = []
        axis_y = []

        # calc a simple demand curve
        for x in range(10):
            if ele == 'perfeita_inelastica':
                axis_x.append(7)
            elif ele == 'perfeita_elastica':
                axis_x.append(x*1000)
            else:
                axis_x.append(x)
            axis_y.append(8 - float(2*x))

        trace = go.Scatter(
            x=axis_x,
            y=axis_y
        )
        traces.append(trace)

        # calc a elasticity curve
        new_axis_y = []
        new_axis_x = []
        for x in range(9):
            if x % 2 == 1:
                continue
            new_axis_x.append(x)
            elasticity = float(((axis_x[x] - axis_x[x+1])/ (axis_x[x]+axis_x[x+1])/2.0)) \
                / float(((axis_y[x] - axis_y[x+1])/ (axis_y[x] + axis_x[x+1]/2.0)))

            new_axis_y.append(elasticity)

        trace = go.Scatter(
            x=new_axis_x,
            y=new_axis_y

        )
        traces.append(trace)

        plotly.offline.plot({
            "data": [trace for trace in traces],
            "layout": Layout(title="CURVA "+ele.upper() )},

            auto_open=False, filename='elasticidade_'+ele+'.html',
        )

def pib():
    anos_dinheiro = {
        2000 : 100,
        2001 : 110,
        2002 : 90,
        2003 : 105,
        2004 : 120,
    }
    anos_preco = {
        2000 : 0.55,
        2001 : 0.6,
        2002 : 0.7,
        2003 : 0.45,
        2004 : 0.8,
    }
    traces = []

    for tipo in ['preco_unit', 'nominal', 'real']:
        axis_y = []
        axis_x = []
        for ano in anos_dinheiro:
            if tipo == 'nominal':
                axis_x.append(ano)
                axis_y.append(anos_dinheiro[ano])
            if tipo == 'preco_unit':
                axis_x.append(ano)
                axis_y.append(anos_preco[ano])
            if tipo =='real':
                axis_x.append(ano)
                axis_y.append(anos_dinheiro[ano]/anos_preco[ano] \
                    * anos_preco[2000])

        trace = go.Scatter(
            x=axis_x,
            y=axis_y,
            name = tipo
        )
        traces.append(trace)


    plotly.offline.plot({
        "data": [trace for trace in traces],
        "layout": Layout(title="CURVA PIB" )},

        auto_open=False, filename='pib.html',
    )

def inflacao():
    anos_dinheiro = {
        2000 : [100, 100],
        2001 : [150, 180],
        2002 : [155, 182],
        2003 : [165, 190],
        2004 : [166, 193],
    }
    percent = [0.6, 0.4]
    traces = []

    axis_y = []
    axis_x = []
    base = anos_dinheiro[2000][0] * percent[0] + anos_dinheiro[2000][1] * percent[1]
    for ano in anos_dinheiro:
        axis_x.append(ano)
        tot = anos_dinheiro[ano][0]*percent[0] + anos_dinheiro[ano][1]*percent[1]
        axis_y.append((tot - base)/100)

    trace = go.Scatter(
        x=axis_x,
        y=axis_y,
    )

    plotly.offline.plot({
        "data": [trace],
        "layout": Layout(title="CURVA INFLACAO" )},

        auto_open=False, filename='inflacao.html',
    )



def gini(y):
    "Compute the Gini coefficient (a measure of equality/inequality) in a population, y."
    y = sorted(y)
    n = len(y)
    numer = 2 * sum((i+1) * y[i] for i in range(n))
    denom = n * sum(y)
    return (numer / denom) - (n + 1) / n

def hist(population, label='pop', **kwargs):
    "A custom version of `hist` with better defaults."
    label = label + ': G=' + str(round(gini(population), 2))
    h = plt.hist(list(population), bins=30, alpha=0.5, label=label, **kwargs)
    plt.xlabel('wealth'); plt.ylabel('count'); plt.grid(True)
    plt.legend()

def random_split(A, B):
    "Take all the money uin the pot and divide it randomly between the two actors."
    pot = A + B
    share = random.uniform(0, pot)
    return share, pot - share
def anyone(N): return random.sample(range(N), 2)

def simulate(population, T, transaction=random_split, interaction=anyone):
    "Run simulation on population for T transactions; yield (t, pop) at each time step."
    population = population.copy()
    yield population
    for t in range(1, T + 1):
        i, j = interaction(len(population))
        population[i], population[j] = transaction(population[i], population[j])
        yield population

import statistics

def show(population, k=40, percentiles=(1, 10, 50, 90, 99), **kwargs):
    "Run a simulation for k*N steps, printing statistics and displaying a plot and histogram."
    N = len(population)
    start = list(population)
    results = [(t, sorted(pop)) # Sort results so that percentiles work
               for (t, pop) in enumerate(simulate(population, k * N, **kwargs))
               if t % (N / 10) == 0]
    times = [t for (t, pop) in results]
    # Printout:
    print('   t    Gini stdev' + (' {:3d}%' * len(percentiles)).format(*percentiles))
    print('------- ---- -----' + ' ----' * len(percentiles))
    fmt = '{:7,d} {:.2f} {:5.1f}' + ' {:4.0f}' * len(percentiles)
    for (t, pop) in results:
        if t % (4 * N) == 0:
            data = [percent(pct, pop) for pct in percentiles]
            print(fmt.format(t, gini(pop), statistics.stdev(pop), *data))
    # Plot:
    plt.hold(True); plt.xlabel('wealth'); plt.ylabel('time'); plt.grid(True)
    for pct in percentiles:
        line = [percent(pct, pop) for (t, pop) in results]
        plt.plot(line, times)
    plt.show()
    # Histogram:
    R = (min(pop+start), max(pop+start))
    hist(start, 'start', range=R)
    hist(pop, 'end', range=R)

def percent(pct, items):
    "The item that is pct percent through the sorted list of items."
    return items[min(len(items)-1, len(items) * pct // 100)]

def simulacao():
    import random

    N  = 5000 # Default size of the population
    MU = 100. # Default mean of the population

    population = [random.gauss(mu=MU, sigma=MU/5) for actor in range(N)]


if __name__ == '__main__':

    print 'Escolha o modulo:'
    print '1 - Oferta e Demanda '
    print '2 - Elasticidade'
    print '3 - PIB real e PIB nominal'
    print '4 - Inflacao'

    choice = raw_input()

    if int(choice) == 1:
        oferta_demanda()
    elif int(choice) == 2:
        elasticidade()
    elif int(choice) == 3:
        pib()
    elif int(choice) == 4:
        inflacao()



