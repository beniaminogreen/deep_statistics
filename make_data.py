import numpy as np
import numba
import tqdm
import pickle as pkl

def generate_dataset(n):
        mean = np.random.uniform(-200,200)
        variance = np.random.uniform(.5,30)

        dataset = np.random.normal(mean , variance, n)

        dataset = dataset[:,np.newaxis]

        return(dataset)



for i in range(100):
    datasets = []
    statistics = []
    for _ in tqdm.tqdm(range(500)):
        dataset = generate_dataset(200)

        datasets.append(dataset)
        statistics.append(np.var(dataset) + np.mean(dataset))

    with open(f"data/{i}.pkl", "wb") as f:
        pkl.dump([datasets, statistics], f)

print(statistics)
