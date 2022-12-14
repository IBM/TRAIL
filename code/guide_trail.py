#!/usr/bin/env python
# Quick hack to support Beam Search
from __future__ import annotations # for -> Term
import sys, os, time
import itertools
import dataclasses
from typing import Tuple, Any, Set
import timeout_decorator
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Sequence
from typing import TextIO

def open_commands(dir, writeCommands):    
    commands = None
    command_responses = None
    if dir:
        pwd = os.getcwd()
        os.chdir(dir)
    print('opening commands for dir ', dir)
    try:
        if writeCommands:
            print('GUIDE BEAM?');         sys.stdout.flush()            
            commands = open("commands", "w")
            print('WRITING to "commands"')            ;         sys.stdout.flush()
            command_responses = open("command_responses")
            print('READING from "commands_responses"');         sys.stdout.flush()
        else:
            print('USE BEAM?');         sys.stdout.flush()
            commands = open("commands")
            print('READING from "commands"')            ;         sys.stdout.flush()
            command_responses = open("command_responses", "w")
            print('WRITING to "commands_responses"');         sys.stdout.flush()            
    except IOError:
        if commands:
            if not command_responses:
                print("only 'commands' exists! where is 'command_responses'?")
                sys.exit(1)
        else:
            print('NOT reading from "commands"')
        sys.stdout.flush()
    if dir:
        os.chdir(pwd)
    return commands, command_responses

@timeout_decorator.timeout(60)
def readline_timeout(c):
    return c.readline().strip()

# This is a version of waitForExit that only runs maxrunning procs at a time.
# I'm sure there must be a way to do this with some python library, but I'll use my own code for now.
def waitForExitCont(nodes, maxrunning):
    print(f'running {len(nodes)} Trail nodes, max {maxrunning} at a time')
    nwait=0
    nstarted=0
    running=set([])
    for n in nodes:
        print('waitForExitCont',n.trajectory)
        # (n,(pid,path)) in ((n, n.forkcont()) for n in nodes): # note: this is a generator expression, NOT a list
        running.add((n,n.forkcont()))
        nstarted+=1

        while len(running) >= maxrunning or nstarted==len(nodes):
            # exited = [pid for (n, (pid, path)) in running if not os.path.isdir(f"/proc/{pid}")]
            exited = [(n,(pid,path)) for (n, (pid, path)) in running if not os.path.isdir(f"/proc/{pid}")]
            while not exited:
                time.sleep(1)
                nwait+=1
                exited = [(n,(pid, path)) for (n, (pid, path)) in running if not os.path.isdir(f"/proc/{pid}")]

            for (n,(pid,path)) in exited:
                outcome, beam_info = getrval(path)
                TrailInstance.outcome[n.trajectory] = outcome
                print('waitForExitCont exited!', pid, path, outcome)
            running.difference_update(exited)

            if nstarted < len(nodes) or not running:
                break
            print('waitForExitCont still running', len(running), nstarted)
    print('waitForExitCont all procs exited')


def waitForExit(pids):
    print(f'waiting for Trail to exit (pids {pids})')
    #     nexited=0
    nwait = 0
    #     npids=len(pids)
    running = set(pids)
    while running:
        exited = [pid for pid in running if not os.path.isdir(f"/proc/{pid}")]
        if exited:
            print('exited!', exited)
        else:
            time.sleep(1)
            nwait += 1
        running.difference_update(exited)
    print('all procs exited', pids)

# don't create or even look at these
@dataclasses.dataclass
class TrailInstanceRuninfo:
    commands: TextIO
    command_responses: TextIO
    pid: str
    exited: bool    = False
    destroyed: bool = False
    recreated: bool = False

# don't create these directly; create via makeroot() and step()
# hashable, since frozen
@dataclasses.dataclass(frozen=True)
class TrailInstance:
    # allprocs:Dict[Sequence[Any],TrailInstance] = {}  # traj -> TrailInstance
    # path2traj:Dict[str,Sequence[Any]] = {} # string -> traj
    # outcome:Dict[Sequence[Any],str] ={} # traj -> String
    # _destroyed: Dict[TrailInstance,bool] = {}
    # _exited:    Dict[TrailInstance,bool] = {}
    # _recreated: Dict[TrailInstance,bool] = {}
    # https://stackoverflow.com/questions/61937520/proper-way-to-create-class-variable-in-data-class
    #    To create a class variable, annotate the field as a typing.ClassVar or not at all.
    # This doesn't seem to match the official doc (https://docs.python.org/3/library/dataclasses.html#class-variables)
    # but it seems to be true.
    allprocs    = {}  # traj -> TrailInstance
    procruninfo = {}  # TrailInstance -> TrailInstanceRuninfo
    path2traj   = {} # string -> traj
    outcome     = {} # traj -> String

    # instance variables
    path: str # this is used by beam_search to decide ancestry
    trajectory: Tuple[int]
    solvedByThisStep: bool
    trajectory_probs: Tuple[float]
    trajectory_product: float
    selected_actions_hash: str # contents of allselhashhash.txt

    # for destroy/recreate
    parent: TrailInstance
    rank: str # currently just last element of trajectory
    step: str

    @staticmethod
    def trajpath(traj):
        # assert type(traj) == tuple, type(traj)
        path = "/".join([str(n) for n in traj])
        if not path:
            path = "."
        # print('trajpath',traj,path)
        return path

    @staticmethod
    def makeroot():
        c, r = open_commands(".", True)
        c.write("getpid\n");
        c.flush()
        pid = r.readline().strip()
        return TrailInstance.__make__(None, c, r, tuple([]), False, pid, 1.0, -1, -1)

    # @staticmethod
    # def _testsuccess(node):
    #     rval = node.forkcont()

    # don't call this directly
    @staticmethod
    def __make__(parent, c, r, traj, solvedByThisStep, pid, prob, rank, step):
        if not isinstance(traj, tuple):
            sys.exit('traj must be a tuple', traj)
        path = TrailInstance.trajpath(traj)

        if traj in TrailInstance.allprocs:
            sys.exit('attempting to create a TrailInstance that already exists, for ', traj)

        if parent:
            ts = tuple(parent.trajectory_probs + tuple([prob]))
            pprod = parent.trajectory_product * prob
        else:
            ts = tuple([prob])
            pprod = prob
        if not path:
            hsh = "roothash"
        else:
            try:
                with open(path + "/allselhashhash.txt", "r") as f:
                    hsh = f.read().strip()
            except Exception as e:
                hsh = 'oops' + path  # hopefully unique
        runinfo = TrailInstanceRuninfo(c, r, pid)
        rval = TrailInstance(path, traj, solvedByThisStep, ts, pprod, hsh, parent, rank, step)
        print('selhash:', rval.selected_actions_hash)
        if len(TrailInstance.allprocs) > 10000:
            sys.exit('TOO MANY PROCS! You may increase the limit, but as some point the OS will lock you out')
        TrailInstance.allprocs[traj] = rval
        TrailInstance.path2traj[path] = traj
        TrailInstance.procruninfo[rval] = runinfo
        return rval

    def score(self):
        return self.trajectory_product

    def solved(self):
        # if self.trajectory not in TrailInstance.nodessolved:
        #     (pid,path) = TrailInstance._testsuccess(self)
        #     waitForExit([pid])
        #     outcome, beam_info = getrval()
        #     TrailInstance.nodessolved[self.trajectory] = outcome=="SOLVED"
        return self.solvedByThisStep or TrailInstance.outcome[self.trajectory]=="SOLVED"
    
    # auxiliary routine, don't call directly
    def wc(self,s):
        ri = TrailInstance.procruninfo[self]
        if ri.destroyed:
            sys.exit(f'writing command to destroyed instance: {self.path}')
        commands = ri.commands
        try:
            print('WRITING COMMAND',s, "to", self.path); sys.stdout.flush()
            # os.system(f"ls {self.path}")
            commands.write(s);
            commands.write("\n");
            commands.flush() # important
        except:
            print('SOME ERROR!', ri.pid)
            # os.system('ps ux')
            raise

    def rc(self):
        ri = TrailInstance.procruninfo[self]
        command_responses = ri.command_responses

        x = command_responses.readline().strip();
        print(f'READ >{x}<', flush=True)
        # while not x:
        #     print('reading again!', flush=True) # don't know why this sometimes happens
        #     x = self.command_responses.readline().strip();
        #     print(f'READ >{x}<', flush=True)
        return x

    # def getpid(self):
    #     self.commands.write("getpid\n"); self.commands.flush()
    #     pid = self.command_responses.readline().strip()
    #     return pid

    def cont(self):
        ri = TrailInstance.procruninfo[self]
        commands = ri.commands
        print('start runUntilExit', self.path)
        pid = ri.pid
        commands.write("runUntilExit\n"); commands.flush()
        print('start runUntilExit pid', pid)
        return waitForExit(pid, self.path)


    ugly_fork_hack = False

    # this only forks the parent into a new dir;
    # the child doesn't do anything yet.
    def _basicfork(self, dir):
        s=f'fork {dir}'
        forkdir = self.path+f"/{dir}"
        assert not os.path.isdir(forkdir)
        print('FORK COMMAND',s); sys.stdout.flush()

        if not TrailInstance.ugly_fork_hack:
            os.mkdir(forkdir)  # if it fails we exit
            os.mkfifo(forkdir + "/" + "commands")  # this must agree with launch-ee.sh/guide_trail.py
            os.mkfifo(forkdir + "/" + "command_responses")  # this, too
            print('fDIR', forkdir, os.getcwd(), os.listdir(forkdir))

        ri = TrailInstance.procruninfo[self]
        commands = ri.commands
        command_responses = ri.command_responses

        commands.write(s);
        commands.write("\n");
        commands.flush()
        print('WROTE COMMAND'); sys.stdout.flush()
        x = command_responses.readline().strip(); print('READ', x, flush=True)
        ok,pid = x.strip().split(' ')
        if ok!='OK':
            sys.exit(f'BAD FORK0! {forkdir} {x}')

        # I suspect that if we write commands before the child opens it, the value will be lost;
        # so use the existence of 'inited' to signal completion.  ugh.
        # while not os.path.isfile("inited"):
        #     print('waiting for inited', flush=True)
        #     time.sleep(0.001)
        # c,r = open_commands(forkdir, True)
        c = open(forkdir + "/commands", "w")
        print('opened "commands"', flush=True)
        c.write('startup\n'); c.flush()
        print('wrote startup', flush=True)

        r = open(forkdir + "/command_responses", "r")
        print('opened "command_responses"', flush=True)
        x = r.readline().strip();
        print('read init:', x, flush=True)

        x = r.readline().strip();
        ok,epid = x.strip().split(' ')
        if ok!='OK':
            sys.exit('BAD FORK1!')
        print(f'basic fork done {pid} {epid}', flush=True)
        return c,r,pid,epid

    def step_traj(self,step):
        return self.trajectory + tuple([step])

    # now running into mysterious behavior
    @timeout_decorator.timeout(60)
    def _forkstep_part0(self, rank, step):
        rank = str(rank)
        step = str(step)
        # rank is used for the path; the idea is to use small values for conciseness
        # step is the actual Trail index of the clause
        traj = self.step_traj(rank)
        path = TrailInstance.trajpath(traj)
        # print('forkstep',rank,step,traj,path)
        if os.path.isdir(path):
            print(f"This traj exists! {traj}")
            print(f"This traj exists! {traj}",file=sys.stderr)
            assert 0
            # ??? what was this???
            rval = TrailInstance.__make__(c, r, traj, solved, pid)
            assert rval
            return rval
        c, r, pid, epid = self._basicfork(str(rank))
        # stepc='step' if absnum else 'stepx'
        stepc = 'step'
        x = c.write(f'{stepc} {step}\n');
        c.flush()
        print('WROTE stepx command', rank, step, path)
        return (c,r,pid,epid,traj,rank,step)


    # now running into mysterious behavior, using timeouts
    def _forkstep_part1(self, c_r_pid_epid_traj_rs):
        (c, r, pid, epid, traj, rank, step) = c_r_pid_epid_traj_rs # can't remember why I did this
        try:
            x = readline_timeout(r)
        except:
            print('TIMEOUT ERROR IN FORKSTEP1!', traj, flush=True)
            os.system(f"kill -9 {pid} {epid}")
            os.system(f"tar czf '{traj}'.tgz {TrailInstance.trajpath(traj)}")
            print('done error cleanup', flush=True)
            return None
        solved = x=='DONE'
        print('READ stepx output', x)
        if x.startswith('STEPPED'):
            prob=float(x.replace('STEPPED',''))
        else:
            prob=0.0
        rval = TrailInstance.__make__(self, c, r, traj, solved, pid, prob, rank,step)
        if solved or x.startswith('STEPPED') or x=='ERROR':
            return rval
        print(f"UNEXPECTED RETURN for stepx: {x}", flush=True)
        sys.exit(1)        
        return rval
        
    # I broke this into two parts with the thought to let part0 run in parallel.
    def forkstep(self, rank, step):
        # absnum True only in random_runs
        stuff = self._forkstep_part0(rank, step)
        if not stuff:
            return None
        return self._forkstep_part1(stuff)

    def maybe_forkstep(self, rank, step) -> TrailInstance:
        traj = self.step_traj(rank)
        if traj in TrailInstance.allprocs:
            return TrailInstance.allprocs[traj]
        return self.forkstep(rank, step)

    def forkcont(self):
        c,r,pid,epid = self._basicfork("cont")
#         c,r = open_commands(self.path+"/cont", True)
#         print('getting pid', flush=True)
#         pid = self.getpid()
        print('start runUntilExit', pid)
        c.write("runUntilExit\n"); c.flush()
        print('start runUntilExit2', pid)
        return (pid, self.path+"/cont")

    def showclauses(self):
        self.wc(f'showclauses')
        return self.rc()

    def nactions(self):
        self.wc('nactions')
        x = self.rc()
        ok,na,npc = x.strip().split(' ')
        return (int(na),int(npc))

    def beam_steps(self,n):
        self.wc(f'beam_steps {n}')
        x = self.rc()
        # NUTS - don't know why but getting DONE message here
        if x=="DONE":
            return []
        hashes = x.strip().split(' ')
        x = self.rc()
        probs = x.strip().split(' ')
        x = self.rc()
        xs = x.strip().split(' ') # OK prob0 prob1 ...
        if len(hashes) != len(xs)-1:
            print('bad beam_steps',hashes,probs,xs)
        assert len(hashes) == len(xs)-1
        return [(int(xs[i+1]),float(probs[i]), hsh) for (i,hsh) in enumerate(hashes)] # this may not actually return n items

    def ping(self):
        self.wc('ping')
        x = self.rc()
        if x!='OK':
            print('BAD PING!')
            sys.exit(1)

    def wait(self):
        self.wc(f'wait')
        x = self.rc()
        if x != "OK":
            # sys.exit(f"bad wait rc: {x}")
            print(f"bad wait rc: {x}")

    def exit(self):
        ri: TrailInstanceRuninfo = TrailInstance.procruninfo[self]
        commands = ri.commands
        command_responses = ri.command_responses

        if ri.exited:
            sys.exit("calling exit on TrailInstance twice:",str(self))
        if self.solvedByThisStep:
            #already exited
            print('exit - solvedByThisStep')
        else:
            try:
                self.wc('exit')
                # rval = self.rc()
            except:
                print('process', self.path, 'already exited!')

        commands.close() # so you will get an error if you try to write another command
        command_responses.close()
        os.unlink(f"{self.path}/commands")
        os.unlink(f"{self.path}/command_responses")
        ri.commands = None
        ri.command_responses = None

        del TrailInstance.allprocs[self.trajectory]
        del TrailInstance.path2traj[self.path]
        # if self.trajectory in TrailInstance.outcome:
        #     del TrailInstance.outcome[self.trajectory]
        ri.exited = True

        if self.parent and TrailInstance.procruninfo[self.parent].commands:
            self.parent.wait()
        # root.wait()

        # return rval

    def destroy(self):
        ri: TrailInstanceRuninfo = TrailInstance.procruninfo[self]
        if ri.destroyed:
            sys.exit("calling destroy on TrailInstance twice:", str(self))
        print('destroying',self.path)
        self.exit()
        # construct list first, because values() will change otherwise
        # slow if traj is deep
        for ch in [n for n in TrailInstance.allprocs.values() if self is n.parent]:
            ch.destroy()
        if self.trajectory: # [] for root
            print(f"destroying dir: rm -rf {self.path}")
            os.system(f"rm -rf {self.path}")
        # del TrailInstance.procruninfo[self]
        ri.destroyed = True

    def recreate(self):
        if not self.trajectory:
            sys.exit("can't recreate the root!")
        ri: TrailInstanceRuninfo = TrailInstance.procruninfo[self]
        if not ri.destroyed:
            sys.exit(f"calling recreate on non-destroyed TrailInstance: {self}")
        if ri.recreated:
            sys.exit(f"calling recreate on TrailInstance twice: {self}")
        # TrailInstance._destroyed.remove(self) # error if not in there
        # TrailInstance._exited.remove(self)
        ri.recreated = True

        pri: TrailInstanceRuninfo = TrailInstance.procruninfo[self.parent]
        if pri.destroyed:
            self.parent.recreate()
        # yuck
        # del TrailInstance.allprocs[self.trajectory]
        # del TrailInstance.path2traj[self.path]
        # del TrailInstance.procruninfo[self]

        # TrailInstance.ugly_fork_hack = True
        rval = self.parent.forkstep(self.rank, self.step)
        # TrailInstance.ugly_fork_hack = False

        if rval != self:
            print(str(self))
            print(str(rval))
            sys.exit("recreate isn't the same!")

        assert TrailInstance.allprocs[self.trajectory] == self
        TrailInstance.allprocs[self.trajectory] = self

        assert TrailInstance.path2traj[self.path] == self.trajectory
        TrailInstance.path2traj[self.path] = self.trajectory

        # use new procruninfo
        # TrailInstance.procruninfo[self]

        return None


# this does NOT use communication channels; Trail should have exited by this point.
# we have to remove files to leave the state clean for the next child
def getrval(path):
    solved=os.path.isfile(f"{path}/episodeResult.gz")
    noproof=os.path.isfile(f"{path}/eprover.no-proof") # see e_prover.py
    print('getrval', solved, f"{path}/episodeResult.gz", os.getcwd())
    # remove other files??
    result = None    
#     if solved:
#         br = "beam_result"
#         if os.path.isfile(br):
#             with open(br, "r") as f:
#                 result = float(f.readline().rstrip())
#                 print(f"read beam_result {result}")  
#             os.remove(f"beam_result")
#         else:
#             print('NO BEAM RESULT!')
#             result = None
#     else:
#         result = None
    return ("SOLVED" if solved else "NOPROOF" if noproof else "TIMEOUT", result)      

# def simple_example(root):
#     # simple test - create a small tree, fully eval the frontier, just 2 procs max
#     for depth in range(2):
#         frontier = [n.forkstep(i) for i in [0,1] for n in frontier]
#         nodes += frontier
#     waitForExitCont(frontier, 2)

def forkchildren(n):
    return [n.maybe_forkstep(i,step) for (i, (step,prob,hsh)) in enumerate(n.beam_steps(3))]

def maybe_waitForExitCont(nodes, mx):
    nodesToRun = [n for n in nodes if n.trajectory not in TrailInstance.outcome]
    print("BEAM wait", len(nodesToRun), len(nodes))
    waitForExitCont(nodesToRun, mx)

# assumes just successful/unsuccessful
def beam_trim(nodes,score, width):
    if len(nodes)<=width:
        return nodes
    # self.rnd = np.random.RandomState(seed=gopts().action_seed if gopts().deterministic_strategy else None)
    # return self.rnd.choice(len(pi), p=pi)  # np.random.choice(len(pi), p=pi)
    successful = [n for n in nodes if score[n]>0.0] # assumes 1.0/0.0 for now
    w0 = len(successful)
    if w0 >= width:
        return random.sample(successful, width)
    failed = [n for n in nodes if not n in successful]
    return successful + random.sample(failed, width - w0)

def beam_search_local(node0, width, depth):
    print(f"BEAM_SEARCH LOCAL", node0.path)
    assert depth>1
    frontier=[node0]
    node0_children=[]
    all_descendents=[]
    for dp in range(depth):
        frontier = [c for n in frontier for c in forkchildren(n)]
        stepx_solved = [n for n in frontier if n.solvedByThisStep] # do NOT used solved()
        if stepx_solved:
            print('BEAM STEPX SOLVED', [n.trajectory[-1] for n in stepx_solved])
            for n in stepx_solved:
                # n.exit()
                pass #  exit() not needed
            frontier = [n for n in frontier if not n.solvedByThisStep]
        all_descendents += frontier
        if not node0_children:
            node0_children=frontier
        print(f"BEAM_SEARCH LOCAL {dp} {len(frontier)}", [n.trajectory[-1] for n in frontier])
        # maybe_waitForExitCont(frontier, 50)
        # score = {n: 1.0 if n.solved() else 0.0 for n in frontier}
        score = {n: n.score() for n in frontier}
        # print('BEAM LSOLVED', [n.trajectory[-1] for n in frontier if n.solved()])
        frontier = beam_trim(frontier, score, width)

    assert width>=len(node0_children)

    max_child=None
    max_child_score=-1.0
    cscores=[]
    tmp=[]
    for c in node0_children:
        cp = c.path + "/"
        cfrontier = [n for n in frontier if n.path.startswith(cp)]
        tmp += cfrontier
        child_score = sum([score[n] for n in cfrontier])
        for n in cfrontier:
            print('BEAM CHILD', c.path, score[n],n.path)
        cscores += [child_score]
        if child_score>max_child_score:
            max_child = c
            max_child_score = child_score
    # tmp: union of (frontier divided by children)
    # should be the same as frontier
    print('BEAM SCORES', cscores, len(tmp), len(frontier))
    if len(tmp) != len(frontier):
        for n in tmp:
            print(n.path)
        print('---')
        for n in frontier:
            print(n.path)
        print('<', list(set(tmp) - set(frontier)))
        print('>', list(set(frontier) - set(tmp)))
    assert len(tmp) == len(frontier)

    # stop all procs that we no longer need
    for child in node0_children:
        if child != max_child:
            cp = child.path + "/"
            for n in all_descendents:
                if n.path.startswith(cp):
                    n.exit()
    return max_child

def beam_search(root, width, depth, overall_depth):
    node0 = root
    for n in range(overall_depth):
        print(f'BEAM_SEARCH', n, len(TrailInstance.allprocs))
        node0 = beam_search_local(node0, width, depth)
        if not node0:
            print('beam_search_local returned None!  stopping')
            return

def random_runs(root):
    assert 0 # hasn't been run in a long time
    nactions, nprocessedclauses = root.nactions()
    print('root actions', nactions,nprocessedclauses)

    traj = []
    rootdepth=0
    # It seems to me that the first few actions don't make much difference,
    # so we just process ALL the initial available actions.
    # Since the ordering doesn't change, it *should* work to simply select item 0 repeatedly.
    for i in range(nactions):
        n = root.forkstep(0, True)  # just use first action, hopefully always one of the original axioms
        root.exit()
        root = n
        rootdepth += 1
    # n.showclauses()
    nodes = [root]
    # expand children with 'rare' clauses (that only appear in <10 frontier nodes)
    depth = 0
    depthincr = 1
    while 1:
        depth += depthincr
        if 1:
            nnewactions = sum([n.nactions()[0] for n in nodes])
            if nnewactions>5000:
                na = 5000 // len(nodes)
                print(f"newactions {depth}: {nnewactions} limiting to {na}")
                frontier = [n.forkstep(i) for n in nodes for i in range(min(na, n.nactions()[0]))]
            else:
                print(f"newactions {depth}: {nnewactions}")
                frontier = [n.forkstep(i) for n in nodes for i in range(n.nactions()[0])]
        frontier = list(filter(None.__ne__, frontier))
        nodes += frontier
        for n in frontier:
            n.showclauses()

        # EPDIR = iter /$ITER / episode /$EPISODE
        # so:
        os.system(f"../../../../scripts/choose-paths.sh {rootdepth} {depth}")

        rare_frontier = []
        with open("paths.txt", "r") as f:
            for pathx in f.readlines():
                path = pathx.rstrip()
                traj = TrailInstance.path2traj[path]
                n = TrailInstance.allprocs[traj]
                rare_frontier.append(n)
                # print(f"frp {len(rare_frontier)} {path}")

        print(f"front {len(nodes)} {len(rare_frontier)} {len(frontier)}")
        print(f"front {len(set(nodes))} {len(set(rare_frontier))}")
        if len(rare_frontier) > 10:
            for n in set(nodes) - set(rare_frontier):
                n.exit()
            nodes = rare_frontier
            if len(nodes) > 100:
                print(f"enough nodes: {len(nodes)}")
                break
        else:
            for n in set(nodes) - set(frontier):
                n.exit()
            nodes = frontier
        print(f"end iter {depth} {len(nodes)}")
        if depth>5:
            print(f"enough depth: {depth}")
            break

    # for n in nodes:
    #     n.exit()
    waitForExitCont(nodes, 50)

@dataclasses.dataclass(frozen=True)
class SearchNode:
    # node: Optional[TrailInstance]
    parent: TrailInstance
    node_ind: int # index into allnodes where this object is
    step_rank: int # small int (say 0..3) giving ranking of this step
    step_ind: int # index of the Trail step to take
    total_cost: float
    hsh: str

def argmin(lst):
  return lst.index(min(lst))

def simple_search(root):
    allnodes = []
    stepped = set([])
    all_hashes = {}
    # costs = np.array([0.0])
    costs = []
    node = root
    live_nodes = set([root])
    live_nodes2 = {}
    for (i, (step_ind, prob, hsh)) in enumerate(node.beam_steps(3)):
        log_prob = -np.log(prob)
        tc = log_prob
        # tc = prob
        node_ind = len(allnodes)
        n = SearchNode(node, node_ind, i, step_ind, tc, hsh)
        print('probs init',node_ind,tc,prob,node.trajectory_probs)
        allnodes.append(n)
        costs.append(tc)

    live_nodes_bound = 2000
    nskipped = 0
    last_report = 0
    report_interval = 10
    least_cost = 0.0
    last_nstepped = 0
    last_report_time = time.time()
    t0 = last_report_time
    timed_out = set([])
    stop_adding_live_nodes = False
    while (time.time() - t0) < 20*60: # len(allnodes)<10*1000:
        if len(allnodes) > last_report + report_interval:
            last_report += report_interval
            not_stepped = set(allnodes) - stepped
            # if stop_adding_live_nodes:
                # I can't figure out how to avoid zombies like this:
                # 31954 pts/4    Z      0:00 [python] <defunct>
                # os.system("bash expdir/scripts/cleanup-dead-procs.sh")

            if not stop_adding_live_nodes:
                shutting_down = (live_nodes - timed_out) - set([n.parent for n in not_stepped])
                for n in shutting_down:
                    print('shutting down',n.path)
                    n.exit()
                    if n!=root:
                        del live_nodes2[n]
                live_nodes -= shutting_down
                if len(live_nodes) > live_nodes_bound:
                    stop_adding_live_nodes = True
            trajs = [len(n.trajectory) for n in live_nodes]
            mintraj = min(trajs)
            maxtraj = max(trajs)
            avgtraj = sum(trajs) / float(len(trajs))
            nmintraj = len([n for n in live_nodes if len(n.trajectory) < mintraj+2])
            # mintn:TrailInstance = next(n for n in live_nodes if len(n.trajectory) == mintraj)
            # maxtn:TrailInstance = next(n for n in live_nodes if len(n.trajectory) == maxtraj)
            tm = time.time()
            dtm = tm-last_report_time
            last_report_time=tm
            dstep = len(stepped) - last_nstepped
            last_nstepped = len(stepped)
            # print('live',len(allnodes),len(live_nodes),len(live_nodes2))
            live_costs = [n.total_cost for n in live_nodes2.values()]
            avg_cost = sum(live_costs)/float(len(live_costs))
            mxcost = max([n for n in costs if n<10000.0])
            print(f"live nodes: {tm-t0:5.0f} {len(live_nodes):5} {len(stepped):5} {len(allnodes):6} {mintraj:2} {maxtraj:2} {avgtraj:5.2f} {nskipped:6} {dtm:5.2f} {dstep/dtm:5.2f} lc: {least_cost:5.2f} {avg_cost:5.2f} {min(live_costs):5.2f} {max(live_costs):5.2f} {mxcost:5.2f} {nmintraj}")
            # ,mintn.trajectory_probs,maxtn.trajectory_probs)
            #set([n.parent for n in stepped])

        # least_i = np.argmin(costs)
        least_i = argmin(costs)
        least_cost = costs[least_i]
        # print('costs',least_cost,costs)
        costs[least_i] = 10000.0 # effectively drop it
        n = allnodes[least_i]
        assert least_cost == n.total_cost
        assert n not in stepped
        stepped.add(n)

        print('stepxx',n.parent.path, stop_adding_live_nodes)
        destroy_new_node = False
        to_destroy = []
        if not stop_adding_live_nodes:
            node = n.parent.forkstep(n.step_rank, n.step_ind)  # node is the TrailInstance that corresponds to n
            print('ida stepped', node.path, n.parent.path, n.step_ind, n.total_cost, n.parent.trajectory_probs,
                  [-np.log(p) for p in n.parent.trajectory_probs])
            live_nodes.add(node)
            live_nodes2[node] = n
        else:
            # step without creating a new live node
            def nodes_to_recreate(nx: TrailInstance):
                if nx in live_nodes:
                    return []
                l = nodes_to_recreate(nx.parent)
                l.append(nx)
                return l

            pns = nodes_to_recreate(n.parent)
            print('recreating', [nx.path for nx in pns])
            for nx in pns:
                nx.recreate()
            print('now stepping')
            node = n.parent.forkstep(n.step_rank, n.step_ind)  # node is the TrailInstance that corresponds to n
            print('ida stepped2', node.path, n.parent.path, n.step_ind, n.total_cost, n.parent.trajectory_probs,
                  [-np.log(p) for p in n.parent.trajectory_probs])
            to_destroy = pns
            destroy_new_node = True

        # for nx in live_nodes:
        #     print('ln',nx.path)
        # print('node',node.path)
        if node.solvedByThisStep:
            with open(node.path + "/eprover.trainpos", "r") as f:
                ntrain = len(f.readlines())
            print("That solved it!", len(node.trajectory), node.path, ntrain)
            break
        bsteps = node.beam_steps(3)
        if not bsteps:
            # HACK - timed out and wrote DONE
            print('timed out',node.path)
            timed_out.add(node)
        for (i, (step_ind, prob, hsh)) in enumerate(bsteps):
            npath = f"{node.path}/{i}"
            if hsh in all_hashes:
                nskipped += 1
                print(f"ida skipped {npath} == {all_hashes[hsh]}")
            else:
                log_prob = -np.log(prob)
                tc = n.total_cost+log_prob
                print('probs',tc,node.trajectory_probs,prob,n.total_cost,log_prob, [-np.log(p) for p in n.parent.trajectory_probs[1:]],sum([-np.log(p) for p in n.parent.trajectory_probs[1:]]))
                # tc = node.trajectory_product
                node_index = len(allnodes)
                print('ida adding',node_index,node.path, i, tc,log_prob,prob, node.trajectory_probs)
                n1 = SearchNode(node, node_index, i, step_ind, tc,hsh)
                allnodes.append(n1)
                costs.append(tc)
                all_hashes[hsh] = npath
        if destroy_new_node:
            print('now destroying', [nx.path for nx in pns])
            if to_destroy:
                to_destroy[0].destroy()
            else:
                print('destroying new node')
                node.destroy()
            destroy_new_node = False
    for n in live_nodes:
        n.exit()
    print('all done!')

# This program should be launched in the episode directory, NOT in the experiment dir
if __name__ == '__main__':
    root = TrailInstance.makeroot()

    print('starting')
    root.ping()
    nactions, nprocessedclauses = root.nactions()
    print('root actions', nactions,nprocessedclauses)

    if 1:
        simple_search(root)
    else:
        # the list of nodes is already in TrailInstance.allprocs, but I maintain this anyway
        nodes = [root]
        frontier = [root]

        # simple_example(root)
        # random_runs(root)
        beam_search(root, 6, 4, 40)

    print('allprocs1', list(TrailInstance.allprocs.keys()))
    # If you don't shut these down before you exit, you'll end up with zombie processes
    # for n in nodes:
    #     n.exit()
    # print('allprocs2', list(TrailInstance.allprocs.keys()))

    if TrailInstance.allprocs:
        print("You didn't stop the following procs!")
        print(list(TrailInstance.allprocs.keys()))
        for p in list(TrailInstance.allprocs.values()):
            p.exit()
    print('all done!')
    sys.exit(0)
