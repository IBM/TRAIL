# TRAIL: A Deep Reinforcement Learning Approach to First-Order Logic Theorem Proving

Automated theorem provers have traditionally relied on manually tuned heuristics to guide how they perform proof search. Deep reinforcement learning has been proposed as a way to obviate the need for such heuristics, however, its deployment in automated theorem proving remains a challenge. TRAIL (Trial Reasoner for AI that Learns) is a system that applies deep reinforcement learning to saturation-based theorem proving. TRAIL leverages (a) a novel neural representation of the state of a theorem prover and (b) a novel characterization of the inference selection process in terms of an attention-based action policy.

## Getting started

### Setup
  To run the TRAIL reasoner you need anaconda with python 3.10.
  
  The following assumes that you have installed anaconda and are in the Trail installation directory.
  Run this script to create the environment and install the required packages.
  READ THE DIRECTIONS BELOW BEFORE EXECUTING THIS.
  ```
  bash scripts/create-conda-env.sh
  ```

At one time several CUDA versions were supported, but currently we only support 11.6.
In principle, you should be able to modify the script to use a different version,
although it may take some experimentation to find a combination of versions that avoids package version conflicts.

IF YOU HAVE TROUBLE, please try the following:
FIRST, remove any existing anaconda installation; THEN install anaconda; THEN create an env; e.g.
  ```
  rm -rf ~/anaconda3
  bash ~/Anaconda3-2022.10-Linux-x86_64.sh -b -p ~/anaconda3
  bash scripts/create-conda-env.sh
  ```


### Run

The following assumes that you set the variable TRAIL_DIR to where this main directory of this repository (e.g. `export TRAIL_DIR=~/Trail`). 

#### Downloading datasets
We used MPTP2078 and M2K datasets from https://github.com/zsoltzombori/plcop. Details about these datasets can be found in our papers [[1](https://ieeexplore.ieee.org/document/9669114), [2](https://ojs.aaai.org/index.php/AAAI/article/view/16780)].
Due to licencing, we could not not include the data itself as part of this repository. However, these are the steps needed to get the two datasets working in TRAIL:

```
git clone https://github.com/zsoltzombori/plcop
cp plcop/theorems/mptp2078b/* $TRAIL_DIR/data/mptp2078b_dataset/Problems/TPTP/
cp plcop/theorems/m2np/* $TRAIL_DIR/data/m2np_dataset/Problems/TPTP
```

The default dataset is set in `tcfg/optdefaults.tcfg`) to `TRAIN_DATA_DIR=~/Trail-data/mptp2078b_dataset/`. This step is not necessary if you edit the configuration file and set TRAIN_DATA_DIR according to where the data is. 
```
mkdir ~/Trail-data
mv $TRAIL_DIR/data/mptp2078b_dataset ~/Trail-data
```
#### Starting a run
To start a training run, you must at least pass an `experiment_dir` (a dir to contain all output in).

  ```
  cd $TRAIL_DIR
  bash scripts/trail.sh experiment_dir
  ```

You can optionally pass a configuration file (with suffix '.tcfg') to modify the defaults. For simplicity, these instructions assume that you are running the experiment in your Trail installation dir. You can actually run the scripts wherever you like.

### Configuration file and some important variables
All Trail config variables (and their default) are documented in `tcfg/optdefaults.tcfg`.
The config file you pass to trail overrides these defaults.

Variables in UPPER CASE are bash variables, all other variables are used by the python code.
You can modify both variables on the command line; this takes precedence over any config file options you pass.
You can also set bash variables using your env, but that may be more error-prone.

By default, training is done using the dataset determined by:
````
   TRAIN_DATA_DIR=$TRAIL_DIR/data/mptp2078b_dataset/

   TRAIN_TPTP_PROBLEMS_DIR=$TRAIN_DATA_DIR/Problems
````
To use another dataset, set TRAIN_DATA_DIR to a different value in your config file. 
    
The CHOOSE_PROBS variable should name a bash function that prints out which problems should be solved.
Various functions are defined in scripts/trail-fns.sh
The default (chooseAllProblems) is to solve all of them, but it can be useful to choose a subset to check for minor python bugs; in that case, you may find this useful:

  ```
  bash scripts/trail.sh experiment_dir CHOOSE_PROBS="'chooseNEasy2078 50'"
  ```
   
This prints 50 problems (of the mptp2078 dataset) that trail solved in the first iteration.
In particular, even if we only give Trail 10 seconds (by setting the config variable TIME_LIMIT),
it can still solve a few:

  ```
  bash scripts/trail.sh experiment_dir TIME_LIMIT=10 CHOOSE_PROBS="'chooseNEasy2078 50'"
  ```

By default, only 15 processors are used (we experimented with a dynamic limit, but finally opted for a small fixed one).
To use a different number, you can do this:
  bash scripts/trail.sh expdir POOL_SIZE=50 

Again, you can put these variable settings in your config file,
but using the env variable syntax is more convenient.


### Experiment directory and its contents
The second parameter is the directory to store all output in; it is removed if it already exists.
The following concerns the files in this directory.

By default, all output is saved to log.txt.

The entire Trail `code` dir is copied to this dir, so once a run has started, it is unaffected by any changes you make to your source files.  
The dir is intended to be completely self-contained.

The best way to see results is to use the report.sh script:
  ```
  bash scripts/report.sh experiment_dir
  ```

Problems in the input data dir are assiged numeric ids by sorting the file names alphabetically;
the result is the 'problems' dir.
Each iteration has its own dir, e.g. iter/1.
Inside the iteration dir are dirs for running episodes ('e1' etc), reports ('r'), and training ('g1' etc).
The integer suffixes are for restarting a phase if it fails.
Inside the 'r' dir is a file 'solved.txt' which contains the ids of the problems solved.

iter/N/r/selfplay_train_sum.tsv contains a summary of the TRAIL's performance at each iteration w.r.t. completition ratio, average scores compared to E ... etc


### Provers
Trail uses a modified version of eprover that is checked out and used automatically, so you don't need to build it yourself
(assuming you are using a linux system).
In general, you should never need to use a different version.  
From time to time, we make changes to the interface between Trail and eprover, so one version of eprover may not work with a different version of Trail; do NOT assume that eprover is the same from version to version.


### Dedicated GPU Server for training:
TRAIL requires a GPU for training.
If you use CCC (internal cluster with a job scheduler) or a machine that has a dedicated GPU, the system will work as desired using the defaults.
At one time we were able to do GPU training remotely, but with the move to CCC that has been dropped.

### Subjobs
When eprover is called with the --auto-schedule option, it tries up to ten different strategies in sequence.
The modified version of eprover that comes with Trail allows Trail to select any one of these strategies
using the environment variable TRAIL_AUTO.
Just as eprover's auto-schedule mode tries 10 different strategies rather than use just one the entire time,
Trail performs better if it runs for a shorter time with several models trained on different auto-schedule modes.
The easiest way to implement this was to essentially do several independent runs inside the main Trail directory.
An example of this is tcfg/32batch_journal_whole_graph_9epochs_25secs.tcfg.
Each of these independent jobs is called a 'subjob'.
Each subjob has its own subdir (e.g. 'a0') in the main directory.
Other than that, the directory structure is the same.
The reporting tool (scripts/report.sh) combines the results of the subjobs, giving both a combined report and
reports for each of the subjobs.

### Demo
Included in the installation are trained models for mptp2078.
These models are the result of training 39 iterations with four different settings for eprover (as described in section 'Subjobs').
Assuming you are in your installation dir, you should be able to execute:

  ```
  bash scripts/launch-model39.sh &
  ```

This only runs the first iteration, with GPU training, so it should work on clusters without a GPU.
The launch-model39.sh script calls the report.sh script to show the results.

## Paper 

If you use TRAIL in your research, please cite our papers:

```
@article{abdelaziz2022learning,
  title={Learning to guide a saturation-based theorem prover},
  author={Abdelaziz, Ibrahim and Crouse, Maxwell and Makni, Bassem and Austel, Vernon and Cornelio, Cristina and Ikbal, Shajith and Kapanipathi, Pavan and Makondo, Ndivhuwo and Srinivas, Kavitha and Witbrock, Michael and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
@inproceedings{crouse2021deep,
  title={A deep reinforcement learning approach to first-order logic theorem proving},
  author={Crouse, Maxwell and Abdelaziz, Ibrahim and Makni, Bassem and Whitehead, Spencer and Cornelio, Cristina and Kapanipathi, Pavan and Srinivas, Kavitha and Thost, Veronika and Witbrock, Michael and Fokoue, Achille},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={7},
  pages={6279--6287},
  year={2021}
}
```


