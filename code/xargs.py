#!/usr/bin/env python
import subprocess, os, sys, signal, time
from subprocess import Popen
import itertools 
from game.dfnames import * 
import glob
from typing import Tuple, Dict,List, Any, Set

# I originally thought we could just use linux xargs for this...
# could use GNU parallel, but I can be more aggressive, since I know how the processes behave
# on trail10 the main problem was running out of memory; that presumably isn't such an issue on CCC.
# now the main use is killing/restarting timed-out episodes.

sleep_tm = 0.5

def get_mem()->Tuple[int,int]:
    #https://superuser.com/questions/980820/what-is-the-difference-between-memfree-and-memavailable-in-proc-meminfo
    # MemFree: The amount of physical RAM, in kilobytes, left unused by the system.
    # MemAvailable: An estimate of how much memory is available for starting new applications, without swapping. 
# MemTotal:       791189828 kB
# MemFree:        51819188 kB
# MemAvailable:   773248564 kB
    with open("/proc/meminfo", "r") as f:
        [_, kb, _] = f.readline().split()  # MemTotal
        tot_mem = int(kb) # // (1024*1024)
        f.readline() # MemFree
        [_, kb, _] = f.readline().split()  # MemAvailable
        mem_in_use = tot_mem - int(kb) # // (1024*1024)
        return tot_mem, mem_in_use


def pdone(proc: Popen)->bool:
    pl = proc.poll()
    #     print('poll ', proc.pid, pl)
    if pl != None:  # careful - if pl:  doesn't work, since Null also evaluates to false
        #         print('poll ', proc.pid, pl)
        return True
    return False

if __name__ == '__main__':
    all_args:List[str]=sys.argv[1:]
    slowGBX, critGBX, maxprocsX, launch, iterX, time_limit_str, episodeDirX = all_args
    maxprocs = int(maxprocsX)
    epids = [(s.strip()) for s in sys.stdin.readlines()]
    tot_mem, mem_in_use = get_mem()
    slowGB = int(slowGBX)
    critGB = int(critGBX)
    time_limit = int(time_limit_str)
    print(slowGB, critGB, launch, iterX)

    print('run_procs', slowGB, critGB)
    time0 = time.time()
    last_tm = time0

    if slowGB >= critGB:
        sys.exit(f"we require slowGB < critGB ({slowGB} < {critGB})")
    slowKB = slowGB*1024*1024
    critKB = critGB*1024*1024
#     print(tot_mem, mem_in_use)
    
    numEps = len(epids)
#     launch = f"{trail_dir}/launch-ee.sh"
    nstarted:int = 0
    suspended:List[Popen] = []
    running:List[Popen] = []
    M=1024*1024
    iter_num = 0
    nfinished_line=0
    totfinished=0
    totsolved=0
    proc_epid:Dict[Popen,int] = {}
    proc_starttm:Dict[Popen,float] = {}
    restarted:Set[int] = set([])
    errmsgs:List[int]=[]

    global epdirX
    epdirX = episodeDirX
    # def episodeDir(epid:int)->str:
    #     # nonlocal epdirX # mypy bug - won't find epdirX
    #     global epdirX
    #     return f"{epdirX}/{epid}"
    def episodeDir(epid:int)->str:
        # python error
        # nonlocal episodeDirX # type: ignore[misc]
        global epdirX
        return f"{episodeDirX}/{epid}" # type: ignore[name-defined]

    def printerrs(errmsgs:List[int])->None:
        for epid in errmsgs:
            for pat in ["Error","Exception"]:
                errpat = f"{episodeDir(epid)}/*{pat}*"
                if glob.glob(errpat):
                    # some errors have very large strings; use 'cut' to limit this
                    os.system(f"echo {epid}; tail -n 20 {errpat} | cut  -c -200")

    last_sleep_tm = time.time()
    epsDone = os.fdopen(3, "w") # 3>$idir/epsDone
    completed = os.fdopen(4,"w")
    while epids or running:
        signal.alarm(60) # at least find out where we weresig if we hang
        changed = False
        _, mem_in_use = get_mem()
        
        finished = [proc for proc in running if pdone(proc)]
        finished_set = set(finished)
        running = [proc for proc in running if not proc in finished_set]

        iter_num+=1
#         if iter_num%20==0:
#             print(f"    {'time':6}:  {'MU':3}  {'slo':3} {'crt':3}   {'strt':4} {'eps':4}   {'run':3} {'ssp':3}   {'finished'}")      
#         print(f"mem {time.time()-time0:6.0f}:  {mem_in_use//M:3d}  {slowGB:3} {critGB:3}   {nstarted:4} {len(epids):4}   {len(running):3d} {len(suspended):3d}   {[proc.pid for proc in finished]}")

        # above slow:  just don't start new ones
        # above crit:  suspend procs
        
#         print('miu', mem_in_use < slowKB, epids, not suspended, len(running) + len(suspended) < maxprocs)
        if mem_in_use < slowKB and epids and not suspended and len(running) + len(suspended) < maxprocs:
            #mypy won't let me use None in repeat()
            for _ in itertools.repeat(1234, len(finished[3:]) if finished and len(finished) > 3+5 else len(epids) if numEps <= maxprocs else 5):
                if epids:
                    id = epids.pop()
                    epdir = episodeDir(id)
                    try:
                        os.mkdir(epdir)
                    except FileExistsError:
                        pass
                    # x=f"{episodeDirX}/{id}"
                    # print('exp',x)
                    proc = subprocess.Popen(["/bin/bash", launch, iterX, time_limit_str, str(id)],
                                            env=dict(os.environ, EXPERIMENT_DIR=os.getcwd(), EPDIR=f"{episodeDirX}/{id}"),
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL
                                        )
                    tm = time.time()
                    # print('started', proc.pid, tm)
                    running.append(proc)
                    proc_epid[proc] = id
                    proc_starttm[proc] = tm
                    nstarted += 1
                    changed = True
        elif mem_in_use < critKB and suspended:
            # resume a proc
            proc = suspended.pop()
            running.append(proc)
            proc.send_signal(signal.SIGCONT)
#             print('resm', proc.pid) #, len(running), len(suspended))
        elif mem_in_use >= critKB and running:
            # suspend a proc
            proc = running.pop()
            suspended.append(proc)
            proc.send_signal(signal.SIGSTOP)
#             print('susp', proc.pid) #, len(running), len(suspended))
#         else:

        if finished:
            changed = True
        # print('xag ', len(finished) , flush=True)
        #         print('.' * len(finished), end='')
        for proc in finished:
            errpat = f"{episodeDir(proc_epid[proc])}/*Error*" # must sync with printerrs...
            someerr = glob.glob(errpat)
            expat = f"{episodeDir(proc_epid[proc])}/*Exception*"
            someex = glob.glob(expat)
            examples=os.path.isfile(f"{episodeDir(proc_epid[proc])}/{proc_epid[proc]}.etar")
            noexamples=os.path.isfile(f"{episodeDir(proc_epid[proc])}/noexamples")
            solved=examples or noexamples
            noproof=os.path.isfile(f"{episodeDir(proc_epid[proc])}/eprover.no-proof") # see e_prover.py
            prevsolved=os.path.isfile(f"{episodeDirX}/{iter_num-1}/{proc_epid[proc]}.etar") # nuts - doesn't work anymore
            epid = proc_epid[proc]
            if solved: 
                totsolved += 1
                epsDone.write(f"{epid}\n")
                epsDone.flush()
            completed.write(f"{epid}\n")

            print(' ' if solved and prevsolved
                    else '+'  if solved and not prevsolved
                    else '-'  if not solved and prevsolved
                    else 'X' if someex
                    else 'E' if someerr
                    else 'N' if noproof
                    else '.', end='')
            last_sleep_tm = time.time()

            nfinished_line += 1
            totfinished += 1
            if someerr or someex:
                errmsgs.append(epid)
                if len(errmsgs)>5:
                    print('')
                    printerrs(errmsgs)
                    if len(errmsgs) == totfinished:
                        sys.exit('Giving up')
                    errmsgs=[]
        
            mxline=50
            if nfinished_line >= mxline:
                print(f" {totsolved:4} {totfinished:4} {100.0*totsolved/totfinished:5.1f}") #    (r={len(running)})")
                nfinished_line=0
                printerrs(errmsgs)
                errmsgs=[]
            
        # procs are hanging on CCC, don't know why
        tm = time.time()
        killed = []
        long_time=(100 if time_limit>10 else 20)+time_limit

        if 1:
            for proc in running:
                rtm = tm - proc_starttm[proc]
                if rtm > long_time:
                    s = "restarting" if proc not in restarted else "killing"
                    id=proc_epid[proc]
                    print(f'\n{s} proc for prob', id, rtm,time_limit, flush=True)
                    proc.kill()
                    killed.append(proc)
                    last_sleep_tm = time.time()
                    if id not in restarted:
                        restarted.add(id)
                        epids.append(id)
                        epdir = episodeDir(id)
                        try:
                            os.rename(epdir, epdir.replace('/episode/', '/failed-episode/'))  # nuts
                        except:
                            print(f"couldn't rename epdir for {id}! {epdir}")

        # print('ps0', flush=True)
        ps = f"vmstat --unit M; top | head -n 5; ps --forest -o pid,stat,time,cmd -g $(ps -o sid= -p {os.getpid()})"
        # print('xargs', len(killed),changed,flush=True)
        if killed:
            last_tm = time.time()
            running = list(set(running) - set(killed))
            print(ps)
            os.system(ps) # I've never actually used this, but perhaps it could help debug issues
        elif changed:
            last_tm = time.time()
        elif False and time.time() - last_tm > long_time:
            print('SOMETHING WRONG - no change for a long time',long_time, flush=True)
            last_sleep_tm = time.time()
            # print(f"ps u {' '.join([proc.pid for proc in running])}")
            # os.system(f"ps u {' '.join([str(proc.pid) for proc in running])}")
            #https://superuser.com/questions/363169/ps-how-can-i-recursively-get-all-child-process-for-a-given-pid
            # os.system(f"ps --forest -o pid,stat,time,cmd -g {os.getpgid(0)}")  # $(ps - o sid= -p {os.getpid()})")
            print(ps)
            os.system(ps)
            for proc in running:
                print('killing proc', proc_epid[proc], tm - proc_starttm[proc])
                proc.kill()

#         if nstarted >= 60:
#             time.sleep(sleep_tm)
        sys.stdout.flush()
        epsDone.flush()
        completed.flush()

        try:
            time.sleep(3 if nstarted < min(60,maxprocs-10) else sleep_tm)
        except Exception as e:
            #??
            print('SLEEP INTERRUPT',repr(e),flush=True)
        # desperation
        if time.time() - last_sleep_tm > 60:
            print('still awake',flush=True)
            last_sleep_tm = time.time()

    if nfinished_line:
        print(f" {totsolved:4} {totfinished:4} {100.0*totsolved/totfinished:5.1f}")
    printerrs(errmsgs)
    print('xargs.py all done!')
    sys.stdout.flush()
    epsDone.close() # close this after flushing stdout; otherwise our stdout gets mixed up with the next phase
    completed.close()
   