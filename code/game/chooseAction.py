from guide_trail import open_commands
from gopts import *
import numpy as np
from utils import ClauseHasher

# This class exists solely to allow us to control execute_episode from an external process.
# If there is a command pipe file, then we read from that;
# otherwise we just choose actions normally using the policy network.
class ChooseAction:

    def __init__(self):
        commands, command_responses = open_commands("", False)
        self.commands = commands
        self.command_responses = command_responses
        self.hack_last_response = command_responses
        seed = gopts().deterministic_randseed
        self.rnd = np.random.RandomState(seed=seed if seed else None)
        self.beam_search = (commands != None)
        self.clause_hasher = ClauseHasher() if gopts().print_clause_hashes or self.beam_search else None
        # self.selected_clauses = [] # duplicate of eprover's
        print('CA INIT', commands == None)

    # NOT done for 'continue'; in that case, guide_trail polls for exit of process.
    # only done in case we stop for 'step' command (SOLVE/TIMEOUT/NOPROOF)
    def wrdone(self):
        if self.hack_last_response:
            print('WRITING DONE')
            self.hack_last_response.write("DONE\n")

    def wr(self, s):
        # do NOT terminate s with a newline! we write it here.
        if self.commands:
            print('WRITING RESPONSE', s)
            sys.stdout.flush()
            self.command_responses.write(s)
            self.command_responses.write("\n")
            self.command_responses.flush()
        #                     x = r.readline(); print('READ', x)
        return

    def rndchoice(self, pi):
        return self.rnd.choice(len(pi), p=pi)  # np.random.choice(len(pi), p=pi)

    # def chooseAction(self, prover, state, pi):
    #     action = self.chooseActionX(prover,state,pi)
    #     self.selected_clauses.append(action[0])
    #     return action

    def chooseAction(self, prover, state, pi):
        if not self.commands:
            # vamp_steps.remove(vamp_steps[0])
            #             rndc_stime = time.time()
            # print(f"sum(pi) = {sum(pi)}")
            return self.rndchoice(pi)
        #             rnd_choice_time += time.time() - rndc_stime

        # rank_actions()
        #             https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
        #             def argsort(seq):
        #                 return sorted(range(len(seq)), key=seq.__getitem__)
        #             ranked_pi = argsort(pi)
        # FIX THIS
        ranked_pi = [self.rndchoice(pi), self.rndchoice(pi), self.rndchoice(pi)]

        action = None
        while not action:
            #                 x = array.array('i')
            #                 x.fromfile(commands, 4)
            print('reading', flush=True)
            line = self.commands.readline().strip()
            print('read line', line, flush=True)
            while not line:
                # sometimes mysteriously I get a blank line; hack for now
                print("DIDN'T GET ANYTHING - try again")
                self.commands.close()
                self.commands = open("commands")
                line = self.commands.readline().strip()
                print('read line', line)
            x = line.split(' ')
            print('got command', x)
            cmd = x[0]
            if len(x) > 1:
                arg = x[1]
                if len(x) > 2:
                    print('too many args!')
                    sys.exit(1)
            x = None  # avoid inadvertent uses

            if cmd == 'exit':
                print('COMMANDED TO EXIT!')
                prover.stop()
                # self.wr('OK')
                sys.exit(0)

            elif cmd == 'wait':
                print('COMMANDED TO WAIT!')
                pid, exit_code = os.wait()
                self.wr('OK')

            elif cmd == 'showclauses':
                prover.showclauses()
                with open("pi", "w") as f:
                    for s in pi:
                        print(str(s), file=f)
                with open("ranked_pi", "w") as f:
                    for s in ranked_pi:
                        print(str(s), file=f)
                self.wr('OK')  # need response to signal completion

            elif cmd == 'fork':  # for just Trail, child cd's into new dir
                # if this command fails, we know that we don't have to kill a new eprover process
                # we can remove the dir and try again
                print('FORKING into', arg)
                # os.mkdir(arg)  # if it fails we exit
                sys.stdout.flush()
                sys.stderr.flush()
                child_pid = os.fork()
                if child_pid > 0:  # this is the parent process
                    print("Parent pid/child pid:", os.getpid(), child_pid, flush=True)
                    self.wr(f"OK {child_pid}")

                    if arg == 'cont':
                        # I don't understand why, but if we don't 'wait' for these processes,
                        # they wind up as zombies.  Others don't seem to, even though we don't wait for them.
                        # pid, status = os.waitpid(child_pid, 0)
                        _, _ = os.waitpid(child_pid, 0)
                else:  # child process
                    # print('DIR00', arg, os.getcwd(), os.listdir(arg), os.listdir('.'))
                    os.chdir(arg)
                    sys.stdout = open('stdout.txt',
                                      'w')  # https://stackoverflow.com/questions/16582194/python-version-of-freopen
                    print("Child process: ", os.getpid(), " cwd: ", os.getcwd(), flush=True)
                    sys.stderr = sys.stdout  # hopefully this is OK

                    # os.mkfifo(arg + "/" + "commands")  # this must agree with launch-ee.sh/guide_trail.py
                    # os.mkfifo(arg + "/" + "command_responses")  # this, too
                    print('DIR0', arg, os.listdir('.'))

                    with open("pid", "w") as f:  # might be handy
                        f.write(f"{os.getpid()}\n")
                    # print('DIRc', os.listdir("."), flush=True)

                    self.commands.close()
                    self.command_responses.close()
                    print('opening commands', flush=True)
                    self.commands = open("commands")
                    line = self.commands.readline().strip()
                    print('read linex', line, flush=True)

                    print('opening command_responses', flush=True)
                    self.command_responses = open("command_responses", "w")
                    print('done opening', flush=True)
                    self.command_responses.write('forked Trail ready!\n')
                    self.command_responses.flush()
                    print('wrote response', flush=True)
                    # print('child continuing', os.listdir("."), flush=True)
                    # Path("inited").touch()
                    self.hack_last_response = self.command_responses

                    # finish child - eprover not forked yet
                    # you MUST NOT give commands to the parent until this is done
                    # print('done opening', flush=True)
                    # prover.createInputOutput(arg)
                    prover.createInputOutput('.')  # we've already chdir'ed
                    # print('DIR1', os.listdir('.'), flush=True)
                    # prover.reopenInputOutput()
                    try:
                        new_eprover_pid = prover.fork(arg)  # the prover hasn't, though
                    except:
                        print('eprover fork failed!')
                        self.wr(f"FAIL")
                        raise
                        sys.exit(1)

                    self.wr(f"OK {new_eprover_pid}")

            elif cmd == 'getpid':
                self.wr(str(os.getpid()))

            elif cmd == 'beam_steps':
                nc = int(arg)
                ranked_pi = np.argsort(pi)  # slow, don't care
                ranked_pi = list(ranked_pi)
                ranked_pi.reverse()
                actions = [state.availableActions[i] for i in ranked_pi]
                # actions = list(ranked_pi[-nc:])
                with open("allselhashprob.txt", "w") as f:
                    for i in ranked_pi:
                        clause = state._getAction(i)
                        f.write(f"{self.clause_hasher.hash_clause(clause)} {pi[i]:8.6f} {self.clause_hasher.clause_str[clause]}\n")
                with open("allpcs.txt", "w") as f:
                    for clause in state.processed_clauses:
                        f.write(f"{self.clause_hasher.hash_clause(clause)} {self.clause_hasher.clause_str[clause]}\n")
                print('BEAM_STEPS', actions) # , pi[actions[0]], pi[actions[1]], pi[0:5])
                if len(actions) < nc:
                    print('not enough actions!', actions)
                # ugh
                ranked_pi = ranked_pi[:nc]
                self.wr(f"{' '.join([str(self.clause_hasher.hash_clauses(state._getAction(i))) for i in ranked_pi])}")
                self.wr(f"{' '.join([str(pi[i]) for i in ranked_pi])}")
                self.wr(f"OK {' '.join([str(x) for x in ranked_pi])}")

            elif cmd == 'runUntilExit':
                #                 if self.i_am_a_child:
                #                     print("child processes aren't supposed to CONTINUE!")
                #                     sys.exit(1)
                print('CONTINUING, no longer reading commands', str(os.getpid()))
                # this write replaces the various writes in the main loop - ugly
                print('DIRz', os.listdir("."), flush=True)

                self.command_responses.close()
                self.command_responses = None
                self.hack_last_response = None

                self.commands.close()
                self.commands = None
                return self.rndchoice(pi)

            elif cmd == 'step':
                action = int(arg)
                print('STEPPING, using action', action, pi[action])
                break

            # elif cmd == 'stepx':
            #     action_rating = int(arg)
            #     ranked_pi = np.argsort(pi)  # slow, don't care
            #     action = ranked_pi[-(1 + action_rating)]
            #     print('STEPPING, using action_rating', action_rating, action, pi[action])
            #     break

            elif cmd == 'nactions':
                # with open('nactions', 'wb') as f:
                #     f.write(f"{len(state.availableActions)} {len(state.processed_clauses)}\n")
                self.wr(f"OK {len(state.availableActions)} {len(state.processed_clauses)}")

            # just for testing
            elif cmd == 'o':  # ord('s'):
                action = self.rndchoice(pi)
                print('STEPPING, using usual action')
                break

            #             elif cmd == 'r': # ord('r'):
            #                 if not self.i_am_a_child:
            #                     print("the parent process can't RETURN!")
            #                     sys.exit(1)
            #                 #data = [random.random()]
            # #                     data = str(rnd.random_sample())
            #                 # I get the same random number every time...
            #                 #print('RETURNING', data)
            #                 data = str(len(state.availableActions))
            #                 print('RETURNING', data)
            #                 self.wr(data)
            # #                     with open('child_rval', 'wb') as f:
            # #                         f.write(struct.pack('f'*len(data), *data))
            #                 prover.stop()
            #                 sys.exit(0)
            elif cmd == "ping":
                print('hi there')
                self.wr('OK')
            else:
                print('BAD COMMAND:', cmd)
                sys.exit(1)
        print('action chosen', cmd, action)
        return action

