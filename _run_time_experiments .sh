#!/bin/bash
    

iter=" 20"
k=" 10"
err=" 0.2 "


python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset phy --k $k  --overwrite --nPortion 0.2
python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset phy --k $k --nPortion 0.4
python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset phy --k $k --nPortion 0.6
python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset phy --k $k --nPortion 0.8
python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset phy --k $k --nPortion 1


python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset cifar10 --k $k  --overwrite --nPortion 0.2
python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset cifar10 --k $k --nPortion 0.4
python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset cifar10 --k $k --nPortion 0.6
python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset cifar10 --k $k --nPortion 0.8
python alpha_experiment_kmeans.py --err $err --numIter $iter --dataset cifar10 --k $k --nPortion 1

python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset phy --k $k  --overwrite --nPortion 0.2
python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset phy --k $k --nPortion 0.4
python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset phy --k $k --nPortion 0.6
python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset phy --k $k --nPortion 0.8
python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset phy --k $k --nPortion 1

python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset cifar10 --k $k  --overwrite --nPortion 0.2
python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset cifar10 --k $k --nPortion 0.4
python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset cifar10 --k $k --nPortion 0.6
python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset cifar10 --k $k --nPortion 0.8
python alpha_experiment_kmedians.py --err $err --numIter $iter --dataset cifar10 --k $k --nPortion 1