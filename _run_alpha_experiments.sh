#!/bin/bash
    

iter=" 20"
k=" 10"

python alpha_experiment_kmeans.py --err 0.1 --numIter $iter --dataset mnist --k $k --overwrite
python alpha_experiment_kmeans.py --err 0.2 --numIter $iter --dataset mnist --k $k
python alpha_experiment_kmeans.py --err 0.3 --numIter $iter --dataset mnist --k $k
python alpha_experiment_kmeans.py --err 0.4 --numIter $iter --dataset mnist --k $k
python alpha_experiment_kmeans.py --err 0.5 --numIter $iter --dataset mnist --k $k

python alpha_experiment_kmeans.py --err 0.1 --numIter $iter --dataset cifar10 --k $k --overwrite
python alpha_experiment_kmeans.py --err 0.2 --numIter $iter --dataset cifar10 --k $k
python alpha_experiment_kmeans.py --err 0.3 --numIter $iter --dataset cifar10 --k $k
python alpha_experiment_kmeans.py --err 0.4 --numIter $iter --dataset cifar10 --k $k
python alpha_experiment_kmeans.py --err 0.5 --numIter $iter --dataset cifar10 --k $k

python alpha_experiment_kmeans.py --err 0.1 --numIter $iter --dataset phy --k $k  --overwrite
python alpha_experiment_kmeans.py --err 0.2 --numIter $iter --dataset phy --k $k
python alpha_experiment_kmeans.py --err 0.3 --numIter $iter --dataset phy --k $k
python alpha_experiment_kmeans.py --err 0.4 --numIter $iter --dataset phy --k $k
python alpha_experiment_kmeans.py --err 0.5 --numIter $iter --dataset phy --k $k



python alpha_experiment_kmedians.py --err 0.1 --numIter $iter --dataset mnist --k $k --overwrite
python alpha_experiment_kmedians.py --err 0.2 --numIter $iter --dataset mnist --k $k
python alpha_experiment_kmedians.py --err 0.3 --numIter $iter --dataset mnist --k $k
python alpha_experiment_kmedians.py --err 0.4 --numIter $iter --dataset mnist --k $k
python alpha_experiment_kmedians.py --err 0.5 --numIter $iter --dataset mnist --k $k

python alpha_experiment_kmedians.py --err 0.1 --numIter $iter --dataset cifar10 --k $k --overwrite
python alpha_experiment_kmedians.py --err 0.2 --numIter $iter --dataset cifar10 --k $k
python alpha_experiment_kmedians.py --err 0.3 --numIter $iter --dataset cifar10 --k $k
python alpha_experiment_kmedians.py --err 0.4 --numIter $iter --dataset cifar10 --k $k
python alpha_experiment_kmedians.py --err 0.5 --numIter $iter --dataset cifar10 --k $k

python alpha_experiment_kmedians.py --err 0.1 --numIter $iter --dataset phy --k $k  --overwrite
python alpha_experiment_kmedians.py --err 0.2 --numIter $iter --dataset phy --k $k
python alpha_experiment_kmedians.py --err 0.3 --numIter $iter --dataset phy --k $k
python alpha_experiment_kmedians.py --err 0.4 --numIter $iter --dataset phy --k $k
python alpha_experiment_kmedians.py --err 0.5 --numIter $iter --dataset phy --k $k
