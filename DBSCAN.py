from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, lit
spark = SparkSession\
        .builder\
        .appName("PythonPi")\
        .getOrCreate()
sc = spark.sparkContext

import os
import time
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axis as axis
from matplotlib.ticker import FixedLocator
import matplotlib.colors as mcolors
from scipy.spatial.distance import euclidean, cdist
from sklearn.datasets import make_blobs
from sklearn.neighbors import KDTree

########################### Step 1. Divide points into grids ###########################
def findBucket(par):
    global steps
    global splitters
    global num_par
    global eps
    buckets = {(i, j): [] for j in range(steps) for i in range(steps)}
    index = [0, 0]
    lr = [(0, steps-1), (0, steps-1)]
    for x in par:
        # Do a binary search
        increment = [0, 0]
        for k in range(d):
            l, r = lr[k][0], lr[k][1]
            while l < r:
                mid = (l+r+1)//2
                if splitters[k][mid] < x[k]:
                    l = mid
                else:
                    r = mid-1
            index[k] = l
            if (l > 0) and (abs(x[k] - splitters[k][l]) <= 1/2 * eps):
                increment[k] -= 1
            elif (l < num_par-1) and (abs(x[k] - splitters[k][l+1]) < 1/2 * eps):
                increment[k] += 1
        # print(f'index: {tuple(index)}')
        buckets[tuple(index)].append(x)
        if increment[0] != 0:
            buckets[(index[0] + increment[0], index[1])].append(x)
            if increment[1] != 0:
                buckets[(index[0] + increment[0], index[1] + increment[1])].append(x)
        if increment[1] != 0:
            buckets[(index[0], index[1] + increment[1])].append(x)
    for i, bucket in buckets.items():
        if len(bucket) > 0:
            yield (i, bucket)

def partition_func(x):
    global num_par
    return (x[0] + x[1]) % num_par

########################### Step 2. DBSCAN on each grid ###########################
class DBSCAN():
    def __init__(self, eps=10, MinPts=3, leaf_size=40, check_boundary_points=False):
        self.eps = eps
        self.MinPts = MinPts
        self.check_boundary_points = check_boundary_points
        self.leaf_size = leaf_size

    def get_neighbors(self, point, points):
        neighbors = []
        for candidate in points:
            # print(f'point and candidate: {point} {candidate}')
            if euclidean(point, candidate) <= self.eps:
                neighbors.append(candidate)
        return neighbors

    def boundary_threshold(self, p, grid):
        global splitters
        x = splitters[0]
        y = splitters[1]
        x_left = abs(p[0] - x[grid[0]]) <= 1/2 * self.eps
        y_left = abs(p[1] - y[grid[1]]) <= 1/2 * self.eps
        if grid[0] + 1 == len(x) or grid[1] + 1 == len(y):
            return x_left or y_left
        x_right = abs(p[0] - x[grid[0] + 1]) < 1/2 * self.eps
        y_right = abs(p[1] - y[grid[1] + 1]) < 1/2 * self.eps
        return x_left or x_right or y_left or y_right

    def get_points(self, data):
        grids = []
        points = []
        boundary_points = []
        for pair in data:
            grid, points_list_iter = pair[0], pair[1]
            grids.append(grid)
            for points_list in points_list_iter:
                points += points_list
                if self.check_boundary_points:
                    boundary_points += [p for p in points_list if self.boundary_threshold(p, grid)]
        return grids, points, boundary_points
    
    def fit(self, points):
        X = np.array(points)
        tree = KDTree(X, leaf_size=self.leaf_size)

        cluster_id = 0
        points_idx = np.arange(len(points))
        cluster_ids = np.zeros(len(points), dtype=int)
        counts = tree.query_radius(X, r=self.eps, count_only=True)
        core_points_idx = points_idx[counts >= self.MinPts]
        processed_idx = []

        for i in core_points_idx:
            if i in processed_idx:
                continue
            cluster_id += 1
            current_idx = np.array([i])
            all_neighbors_idx = [i]
            while current_idx.size > 0:
                neighbors_idx = tree.query_radius(X[current_idx], r=self.eps)
                neighbors_idx = np.unique(np.hstack(neighbors_idx))
                cluster_ids[neighbors_idx] = cluster_id
                current_idx = neighbors_idx[np.in1d(neighbors_idx, all_neighbors_idx, invert=True)]
                all_neighbors_idx += current_idx.tolist()
            processed_idx += all_neighbors_idx
            if len(processed_idx) == len(points):
                break
            # cluster_id += 1
        labels = {p: (id, count) for p, id, count in zip(points, cluster_ids, counts)}
        return labels, cluster_id

    def fit_data(self, data):
        grids, points, boundary_points = self.get_points(data)
        labels, cluster_id = self.fit(points)
        yield labels, cluster_id#, core_boundary_points

########################### Step 3. Send max cluster id from each grid and do prefix sum ###########################
def get_max_cluster_id(data):
    for tup in data:
        for item in tup:
            if type(item) == int:
                yield item

def shift_cluster_id(index, data):
    global cluster_ids_all
    if index == 0:
        s = 0
    else:
        s = cluster_ids_all[index - 1]
    for tup in data:
        for dictionary in tup:
            for p in dictionary:
                cluster_id = dictionary[p][0]
                count = dictionary[p][1]
                if cluster_id == 0:
                    yield p, (cluster_id, count)
                else:
                    cluster_id += s
                    yield p, (cluster_id, count)
            break

########################### Step 4. Aggregate core boundary points cluster ids ###########################
def merge_id_for_point(it):
    arr = list(it)
    arr = np.stack(arr)
    arr = arr[arr[:, 0] != 0]
    if (arr.size > 0) and (arr[:, 1].max() >= 2):
        return [set(arr[:, 0])]
    else:
        return []

def merge_id_for_par(x, y):
    for i in x:
        for j in y:
            if i.intersection(j):
                i.update(j)
                return x
    return x + y

def toset(x):
    return [[{x[0]}], x[1]]

def add(x, y):
    x[0][0].add(y[0])
    x[1] = max(x[1], y[1])
    return x

def update(x, y):
    x[0][0].update(y[0][0])
    x[1] = max(x[1], y[1])
    return x

def dfs(node, index, taken, sets):
    taken[index] = True
    root = node
    for i, item in enumerate(sets):
        if not taken[i] and not root.isdisjoint(item):
            root.update(dfs(item, i, taken, sets))
    return root

def merge_id_for_all(sets):
    cluster_ids = []
    taken = [False] * len(sets)
    for i, node in enumerate(sets):
        if not taken[i]:
            cluster_ids.append(dfs(node, i, taken, sets))
    return cluster_ids

########################### Step 5. Map cluster id to new cluster id according to merged ids ###########################
def map_cluster_id(it):
    global merged_ids_dict
    for pair in it:
        p, cluster_id = pair[0], pair[1][0]
        cluster_id = merged_ids_dict.get(cluster_id, cluster_id)
        yield p[0], p[1], cluster_id

# colormap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#0000FF']
colormap = list(mcolors.CSS4_COLORS.values())[::-1]
def plot(data):
    # global ax
    # global fig
    global splitters
    global data_path
    df_par = pd.DataFrame(data, columns=['X', 'Y', 'label'])
    name = str(df_par.loc[0, ['X', 'Y']].values)
    df_par['label'] = df_par['label'].map(lambda x: colormap[x])

    try:
        fig, ax = plt.subplots(figsize=(20, 15))
        extent = (25, 22)
        plt.xlim(-0.5, extent[0]+1)
        plt.ylim(-0.5, extent[1]+1)
        ax.grid(which='major')
        ax.xaxis.set_major_locator(FixedLocator(splitters[0]))
        ax.yaxis.set_major_locator(FixedLocator(splitters[1]))

        ax.scatter(x=df_par['X'].to_numpy(), y=df_par['Y'].to_numpy(), c=df_par['label'].to_numpy(), cmap='hsv')
        fig.savefig(data_path + f'{name}.jpg', dpi=150)
        # plt.scatter(x=df_par['X'], y=df_par['Y'], c=df_par['label'])
        # plt.savefig(f'{name}.jpg', dpi=150)
        plt.close()
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='Perform DBSCAN on datasets and plot results')
    parser.add_argument('name', type=str)
    parser.add_argument('n_pts', type=int)
    parser.add_argument('path', type=str)
    parser.add_argument('num_par', type=int)
    parser.add_argument('eps', type=float)
    parser.add_argument('MinPts', type=int)
    parser.add_argument('leaf_size', type=int)

    args = parser.parse_args()

    global eps, data_path, num_par, d, steps, splitters, fig, ax, cluster_ids_all, merged_ids_dict

    name = args.name
    n_pts = args.n_pts
    path = args.path
    num_par = args.num_par
    eps = args.eps
    MinPts = args.MinPts
    leaf_size = args.leaf_size

    data_path = path + f'{name}/{n_pts}/'
    file_path = data_path + f'{name}_{n_pts}.txt'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    X, y = make_blobs(n_samples=n_pts, centers=10, n_features=2, random_state=0)
    X = X + [abs(X[:, 0].min()), abs(X[:, 1].min())]
    # np.savetxt(file_path, X, delimiter=' ', fmt='%.2f')
    X = X.tolist()
    df = spark.createDataFrame(X, schema='X: FLOAT, Y: FLOAT')
    df = df.withColumn('delim', lit(' '))
    df = df.select('X', 'delim', 'Y')
    # rdd = sc.parallelize(X)
    # rdd = rdd.map(lambda x: str(x[0]) + ' ' + str(x[1]))
    # df = spark.createDataFrame(rdd, schema='coord: STRING')
    # df.write.option('header', 'false').mode('overwrite').text(file_path)
    df = df.select(concat(*df.columns).alias('data'))
    df.coalesce(1).write.option('header', 'false').mode('overwrite').text(file_path)
    # df.coalesce(1).write.format('text').option('header', 'false').mode('overwrite').save(file_path)
    # df.rdd.map(lambda x: str(x[0]) + ' ' + str(x[1])).saveAsTextFile(file_path)
    # df.write.text(file_path)

    # num_par = 10   # number of partitions/workers
    s = 4  # sample size factor
    d = 2 # number of dimensions

    par = sc.textFile(file_path, num_par)
    par = par.map(lambda l: tuple(l.split()))
    par = par.map(lambda pair: (float(pair[0]), float(pair[1])))

    extent = par.reduce(lambda x, y: (max(x[0], y[0]), max(x[1], y[1])))
    extent = (math.ceil(extent[0]), math.ceil(extent[1]))
    steps = math.ceil(num_par)
    splitters = np.linspace((0, 0), extent, steps, endpoint=False).T.tolist()

    X = np.array(X)
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.scatter(X[:, 0], X[:, 1])
    ax.grid(which='major')
    ax.xaxis.set_major_locator(FixedLocator(splitters[0]))
    ax.yaxis.set_major_locator(FixedLocator(splitters[1]))
    fig.savefig(data_path + f'{name}_{n_pts}.jpg', dpi=150)

    start = time.time()
    # Step 1. Divide points into grids
    rdd1 = par.mapPartitions(findBucket)
    rdd2 = rdd1.groupByKey(partitionFunc=partition_func)

    # Step 2. DBSCAN on each grid
    print('Initializing DBSCAN\n')
    dbscan = DBSCAN(eps=eps, MinPts=MinPts, leaf_size=leaf_size)
    rdd3 = rdd2.mapPartitions(dbscan.fit_data)
    rdd3.cache()

    # Step 3. Send max cluster id from each grid and do prefix sum
    cluster_ids_all = rdd3.mapPartitions(get_max_cluster_id).collect()
    for i in range(1, len(cluster_ids_all)):
        cluster_ids_all[i] += cluster_ids_all[i-1]
    rdd4 = rdd3.mapPartitionsWithIndex(shift_cluster_id)
    print(cluster_ids_all)

    # Step 4. Aggregate core boundary points cluster ids
    merged_ids = rdd4.groupByKey().map(lambda x: merge_id_for_point(x[1])).treeReduce(lambda x, y: merge_id_for_par(x, y))
    # merged_ids = rdd4.combineByKey(toset, add, update).map(lambda x: x[1][0] if x[1][1] >= MinPts else None).treeReduce(lambda x, y: f(x, y))
    merged_ids = [{cluster_id for cluster_id in s if cluster_id != 0} for s in merged_ids]
    merged_ids = merge_id_for_all(merged_ids)
    merged_ids_dict = {id: min(s) for s in merged_ids for id in s}
    print(merged_ids_dict)

    # Step 5. Map cluster id to new cluster id according to merged ids
    rdd5 = rdd4.mapPartitions(map_cluster_id)
    # fig, ax = plt.subplots(figsize=(20, 15))
    # extent = (25, 22)
    # plt.xlim(-0.5, extent[0]+1)
    # plt.ylim(-0.5, extent[1]+1)
    # ax.grid(which='major')
    # ax.xaxis.set_major_locator(FixedLocator(splitters[0]))
    # ax.yaxis.set_major_locator(FixedLocator(splitters[1]))
    rdd5.foreachPartition(plot)

    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()

spark.stop()