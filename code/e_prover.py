import array
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import psutil
import timeout_decorator
from dfnames import dfnames
from gopts import gopts
from state import ProverDelta

from infsupportingfunctions import *
from proofclasses import FullGeneratingInferenceSequence
from reasoner_environment.prover_utils import CachingParser
from utils import ClauseHasher

@dataclass(frozen=True, eq=True)
class EProverInitializationException(Exception):
    final_resolvent: bool
    no_proof_found: bool

class EProver:
    #parse_actions_time = 0
    #parse_processed_clauses_time = 0
    #parse_without_cache = 0
    #parse_with_cache = 0
    receive_state_time = 0
    state_string_processing_time = 0
    register_new_clauses_time = 0
    PROTOCOL = 13 # the script that builds eprover (make.sh) looks for this line; also trail.sh

    def __init__(self, pbd, eprover_pid, episode, clause_hash:Optional[ClauseHasher]):
        self.pbd = pbd
        self.episode = episode
        self.step = 0  # age
        # self.negated_conjecture = []
        # self.ssd = ssd
        self.parser = CachingParser()

        self.processed_clauses = []
        self.processed_clauses_set = set([])
        self.availableActions = []
        self.availableActions_set = set([])
        self.selected_clauses = [] # keep them to adjust trainpos
        self.clause_hasher = clause_hash # to help avoid duplicates while searching

        self.clause2id = {}
        self.all_clauses = []
        self.all_clauses_actions = []
        self.all_cnf = []

        self.time_spent=0

        self.STOP_COMMAND = 12345678 # just has to fit in a 64-bit signed int
        self.FORK_COMMAND = 123456789

        # self.last_removed_actions_positions_left = []  # nuts
        # self.last_removed_pclauses_positions_left = []  # nuts

        # self.all_clauses_preds = {}

        trailInputName  = dfnames().trailInputName
        trailOutputName = dfnames().trailOutputName

        # this will hang if we get something like:
        #    Type mismatch in argument #1 of v1_xboole_0(1): expected $i but got $int
        #    eprover: Type error
        # so use timeout.  have to define a fn to use a decorator...
        print(f'opening {trailInputName}',flush=True)
        @timeout_decorator.timeout(1)
        def openInput(trailInputName):
            return open(trailInputName, 'r')
        try:
            self.trailInput = openInput(trailInputName)
            # self.trailInput = open(trailInputName, 'r')
        except Exception as e:
            print(f"Couldn't open {trailInputName}!",flush=True)
            traceback.print_exc(file=sys.stdout)
            os.system(f"ls {trailInputName}")
            os.system("cat eprover.txt")
            raise TimeoutError("Couldn't connect with eprover!")
             
        print(f"Trail opened {trailInputName}",flush=True)

        line = self.trailInput.readline().strip()
        print(f"FIRST LINE: {line}")
        if line != f"eproverTrail protocol {EProver.PROTOCOL}":
            print('This is the wrong protocol of eproverTrail!  This version of Trail requires protocol',EProver.PROTOCOL)

        self.trailOutput = open(trailOutputName, 'w')
        print(f"Trail opened {trailOutputName}")
        print(self.STOP_COMMAND, file=self.trailOutput, flush=True)
        print(self.FORK_COMMAND, file=self.trailOutput, flush=True)
        print(f"Trail wrote to {trailOutputName}", flush=True)

        self.trailInputBin = open(trailInputName+"Bin", 'rb')
        x = array.array('i')
        x.fromfile(self.trailInputBin, 1)
        stop = x.pop()
        x.fromfile(self.trailInputBin, 1)
        fork = x.pop()
        print(f"Trail read {stop} and {fork} from TrailInputBin", flush=True)
        assert stop == self.STOP_COMMAND
        assert fork == self.FORK_COMMAND
        # do this last, in case we get 'Type error', for e.g. prob 140 funct_1__t33_funct_1.p
        try:
            self.eprover_proc = EProver._get_eprover_proc(eprover_pid)
        except Exception:
            print('The eprover process died! hopefully it just proved the problem immediately.', flush=True)
            final_resolvent, no_proof_found, used_in_proof_list = self.read_final_eprover_output()
            raise EProverInitializationException(final_resolvent, no_proof_found)

    def fork(self, dir):
        self._sendActionRank(123456789)  # FORK_COMMAND in cco_proofproc.c
        print(dir, file=self.trailOutput, flush=True)  # must be the only thing on the line, to simplify error checking

        if dir==".":
            cpid = self.trailInput.readline().strip()  # this should be the new eprover's pid
            print('read cpid2', cpid)
        else:
            self.trailInput.close()
            self.trailOutput.close()

            print('reopening TrailOutput', flush=True)
            self.trailOutput = open(dfnames().trailOutputName, 'w')
            print('opened TrailOutput', flush=True)
            print('hi-from-Trail', file=self.trailOutput, flush=True)
            print('printed to TrailOutput', flush=True)

            print('reopening TrailInput', flush=True)
            self.trailInput = open(dfnames().trailInputName, 'r')
            print('opened TrailInput', flush=True)
            cpid = self.trailInput.readline().strip()  # this should be the new eprover's pid
            print('read cpid2', cpid)

            print('reopening TrailInputBin', flush=True)
            self.trailInputBin = open(dfnames().trailInputName + "Bin", 'rb')
            print('opened TrailInputBin', flush=True)
            x = array.array('i')
            x.fromfile(self.trailInputBin, 1)
            n = x.pop()
            print('read testbin', n, flush=True)

        try:
            self.eprover_proc = EProver._get_eprover_proc(cpid) # shouldn't fail at this point
        except Exception:
            print('The eprover process died! giving up', flush=True)
            raise
        return cpid

    @staticmethod
    def _get_eprover_proc(eprover_pid):
        try:
            # os.system(f"ls /proc/{eprover_pid}")
            with open(f"/proc/{eprover_pid}/comm", "r") as f:
                line = f.readline().strip()
                print(f"making sure this command is 'eprover': '{line}'")
                assert line == f"eprover.p{EProver.PROTOCOL}"
                return psutil.Process(int(eprover_pid))
        except FileNotFoundError:
            print(f">>>>>>>>>>>>>>>> This eprover process doesn't exist {eprover_pid}!  Probably an error. look at eprover.txt <<<<<<<<<<<<<")

            #         if not self.eprover_proc:
            #             os.system("cat eprover.txt") doesn't print anything, don't know why
            #             sys.exit(1) # if we do that in the 'except' routine, it just throws another error - confusing
            raise

    #just for fun
    def showclauses(self):
        with open("clauses", "w") as f:
            for (s,_) in self.availableActions:
                print(s, file=f)
            for s in self.processed_clauses:
                print(s, file=f)

        with open("allclauses", "w") as f:
            #for s in self.all_clauses:
            for s in self.all_cnf:
                print(s, file=f)

    def createInputOutput(self, dir):
        os.mkfifo(dir + "/" + dfnames().trailInputName)
        os.mkfifo(dir+"/"+dfnames().trailOutputName)
        # os.mkfifo(dir+"/"+"newunprocessed") # nuts
        # os.mkfifo(dir+"/"+"newprocessed")
        os.mkfifo(dir + "/" + dfnames().trailInputName+"Bin")
        print('created new trail input/output')
        
    # def reopenInputOutput(self):
    #     self.trailInput.close()
    #     self.trailOutput.close()
    #     print('opengin Tinput', flush=True)
    #     self.trailInput = open(dfnames().trailInputName, 'r')
    #     print('opened Tinput', flush=True)
    #     self.trailOutput = open(dfnames().trailOutputName, 'w')
    #     print('opened Toutput', flush=True)
    #     print('reopened new trail input/output')
    #     cpid= self.trailInput.readline().strip() # this should be the new eprover's pid
    #     print('read cpid2', cpid)
    
    def cpu_time(self):
        # at the point this is called, eprover may have exited.
        # we could be more careful and place this call in the e_prover interface, but it doesn't seem that important.
        return self.time_spent

    def _cpu_time(self):
        # at the point this is called, eprover may have exited.
        # we could be more careful and place this call in the e_prover interface, but it doesn't seem that important.
        try:
            (pr_user, pr_system, _, _,_) = self.eprover_proc.cpu_times()
            self.time_spent = pr_user+pr_system
#         print('eprover time ', eprover_proc.cpu_times(), eprover_proc)
        except psutil.NoSuchProcess:
            print('(eprover already exited, using 0 time')
            self.time_spent = 0
        return self.time_spent
    
    def init_game(self):
        # presaturating = self.trailInput.readline().strip();
        # try:
        #     psat = int(presaturating)
        # except Exception as e:
        #     print(f"couldn't case 'presaturating' info to int! {presaturating}")
        #     raise e
        # solved_immediately=False
        # assert not psat # haven't actually tested this, so if you get this error, just remove this check and make sure the code works.
        # # this could in principle solve the problem immediately; in that case, we'd need to add an extra return value.
        # if psat:
        #     # maybe set solved_immediately=True
        #     pass
        
        print('init_game0')
        new_game_reply = None
        try:
            processed_clauses_str, unprocessed_clauses_str = self.getStateStrings(False,True)

            # print(processed_clauses_str, unprocessed_clauses_str, newunprocessed_clauses_strs, newprocessed_clauses_strs)
            # print(unprocessed_clauses_str)
            # print(newunprocessed_clauses_strs)
#             for x in (set(unprocessed_clauses_str).difference(newunprocessed_clauses_strs)):
#                 print('d1', x)
#             for x in (set(newunprocessed_clauses_strs).difference(unprocessed_clauses_str)):
#                 print('d2', x)
#             assert set(unprocessed_clauses_str) == set(newunprocessed_clauses_strs)
            available_actions = self.fill_available_actions(unprocessed_clauses_str)
            self.step += 1
            if os.environ["DROPDUPS"] == "4":
                available_actions = list(set(available_actions))

            self.episode.init_negated_conjecture()
            processed_clauses = self.fill_processed_clauses(processed_clauses_str)
            axioms = list(set(available_actions).difference(self.episode.negated_conjecture))
            if processed_clauses:
                assert "VRA" not in os.environ

            # E apparently can produce $false as one of the available actions!
            # I suppose it simply doesn't bother to check for this before entering the Saturate:while loop.
            # If we don't detect this, though, the $false clause will vectorize to all 0s, which causes NaNs and lots of trouble.
            for cl in self.all_clauses:
                if not cl.literals:
                    print('solved immediately!')
                    self.stop() # do this so that the proof gets written out, so that this is no different from other cases
                    return [], [], []
            if not os.environ["ALWAYS_SELECT_SINGLE_LIT_ACTION"]:
                assert not processed_clauses # or??
            
            print(f'done init_game {len(axioms)} {len(processed_clauses)} {len(available_actions)}')
            # for (cl, _) in available_actions:
            #     self._findClause(cl)
            self.availableActions = available_actions
            self.availableActions_set = set(available_actions)
            if os.environ["DROPDUPS"] in ["1", "2", "4"]:
                assert len(self.availableActions) == len(self.availableActions_set)
            return axioms, processed_clauses, available_actions
        
        except Exception as exception:
            print(f"Initializing game for E prover failed because of {exception}")
            traceback.print_exc(file=sys.stdout)
            raise exception

    # @timeout_decorator.timeout(60)
    def _sendActionRank(self, action_rank):
        print('sendActionRank', action_rank, "" if gopts().deterministic_randseed else self._cpu_time(), flush=True)
        print(str(action_rank), file=self.trailOutput, flush=True)



    def stop(self):
        self._sendActionRank(self.STOP_COMMAND)

    # 13s of 723
    def _findClause(self, clause):
        # return self.eprover_available_actions.index(clause)
        x = self.eprover_available_actions.index(clause)
        if "FIND2" not in os.environ:
            return x
        try:
            return self.eprover_available_actions.index(clause, x+1)
        except ValueError:
            return x

    #11s of 723
    def mkset(self,xs):
        return set(xs)

    # eprover may exit unexpectedly if it runs out of a resource (via Error).
        # if that happens, we may end up hanging on a read from one of the pipes;
        # since this is an error condition, we'll just time out.
    def runAction(self, clause):
        clause_id = self._findClause(clause)
        self.selected_clauses.append(clause)
        if self.clause_hasher:
            self.clause_hasher.selected_clauses.append(clause)
            assert self.selected_clauses == self.clause_hasher.selected_clauses

            # if len(clause.literals)==1 and type(clause.literals[0].atom.predicate) == EqualityPredicate and self.episode.age[clause]>0 and self.episode.age[clause] < self.step-1:
            #     print('old= ',self.episode.age[clause],self.step, clause)
        action_rank = clause_id 
        selected_clause = (clause_id, clause)  

#         print('runAction', clause, action_rank)  # TRACE
        
        # eptime = self.trailInput.readline().strip()  # this enforces synchronization
        # print(f'ep {eptime}')
        # time.sleep(1)
        if "MEASURE_JUST_EPROVER_FORK_OVERHEAD" in os.environ:
            save_epp = self.eprover_proc
            self.fork(".")
            self._sendActionRank(action_rank)
            _,_ = self.getStateStrings(True)
            self.stop()
            self.eprover_proc = save_epp

        self._sendActionRank(action_rank)

        # if os.environ["CHECK_SELECTED_CLAUSE"]:
        #     s0 = self.trailInput.readline()
        #     sl = (s0[1:-2] if s0[0]=='(' else s0).replace(" ", "").split('|') # -1 is newline; -2 is ')'
        #     sl.sort()
        #     s1l = str(clause).replace(" ", "").split('|')
        #     s1l.sort()
        #     if sl==s1l:
        #         print('selected matches')
        #     else:
        #         # print(f">{s0[-1]}<>{s0[-2]}<>{s0}<")
        #         print('Trail   selected:',s1l)
        #         print('eprover selected:',sl)

        processed_clauses_str, unprocessed_clauses_str = self.getStateStrings()
#         print('SSs', len(processed_clauses_str), len(unprocessed_clauses_str))
        if processed_clauses_str or unprocessed_clauses_str:
            st = time.time()
#             new_available_actions = self.fill_available_actions(newunprocessed_clauses_strs)
#             for x,_ in new_available_actions:
#                 print('naa', self.step, x)

            available_actions = self.fill_available_actions(unprocessed_clauses_str)
            self.step += 1
            processed_clauses = self.fill_processed_clauses(processed_clauses_str)
            t = time.time() -st

            # I'd rather not have to check for this if it never happens
            # for (cl,_) in available_actions:
            #     if not cl.literals:
            #         print("SAW $FALSE in aa")
            #         raise Exception('$FALSE')
            # for cl in processed_clauses:
            #     if not cl.literals:
            #         print("SAW $FALSE in pc")
            #         raise Exception('$FALSE')
                
            # next_state_availableActions_set = self.mkset(available_actions) # for profiling
            # next_state_processed_clauses_set = self.mkset(processed_clauses)
            next_state_availableActions_set = set(available_actions)
            if "NEVER_DROP_PC" in os.environ:
                next_state_processed_clauses_set = self.processed_clauses_set.union(processed_clauses)
            else:
                next_state_processed_clauses_set = set(processed_clauses)

#             assert set(new_available_actions).issubset(next_state_availableActions_set) # checks
#             print('aa0', [self.all_clauses_actions.index(x) for x in available_actions])
            
            # available_actions, processed_clauses, delta_from_prev_state = \
            #     StateDelta.make(self.processed_clauses, self.processed_clauses_set, next_state_processed_clauses_set,
            #                     self.availableActions, self.availableActions_set, next_state_availableActions_set
            #                     # self.last_removed_actions_positions_left, self.last_removed_pclauses_positions_left
            #                     )
            # self.last_removed_actions_positions_left = delta_from_prev_state.removed_actions_positions_left
            # self.last_removed_pclauses_positions_left = delta_from_prev_state.removed_pclauses_positions_left
            # print('aa1', [self.all_clauses_actions.index(x) for x in available_actions])
            if os.environ["DROPDUPS"] in ["1", "2", "4"]:
                assert len(available_actions) == len(next_state_availableActions_set)
            if "NEVER_DROP_PC" in os.environ:
                assert not delta_from_prev_state.removed_processed_clauses

            tm0=self.time_spent
            tm = self._cpu_time()
            
#             if set(new_available_actions) != delta_from_prev_state.new_availableActions:
#                 for (x,_) in set(new_available_actions) - delta_from_prev_state.new_availableActions:
#                     print('x1', self.all_clauses.index(x), x)
#                 for x,_ in delta_from_prev_state.new_availableActions - set(new_available_actions):
#                     print('y1', self.all_clauses.index(x), x)
#                 assert False
            olaa = len(self.availableActions)
            olpc = len(self.processed_clauses)

            prover_delta = ProverDelta(next_state_availableActions_set - self.availableActions_set,
                                       self.availableActions_set - next_state_availableActions_set,
                                       next_state_processed_clauses_set - self.processed_clauses_set,
                                       self.processed_clauses_set - next_state_processed_clauses_set)
            # for (i, (cl,_)) in enumerate(available_actions):
            #     if i not in self.last_removed_actions_positions_left:
            #         self._findClause(cl)
            self.availableActions = available_actions # important!  use the value returned by make()
            self.availableActions_set = next_state_availableActions_set
            self.processed_clauses = processed_clauses  
            self.processed_clauses_set = next_state_processed_clauses_set

            # for (aa,_) in available_actions:
            #     nx = 0
            #     if not any([type(literal.atom.predicate) == EqualityPredicate for literal in aa.literals]):
            #         for pc in processed_clauses:
            #             if any([type(literal.atom.predicate) == EqualityPredicate for literal in pc.literals]):
            #                 pass
            #             else:
            #                 if not (self.all_clauses_preds[aa] & self.all_clauses_preds[pc]):
            #                     nx = nx+1
            #     if nx:
            #         print('nx',self.clause2id[aa], nx,len(processed_clauses))
            laa = len(self.availableActions)
            lpc = len(self.processed_clauses)
            # if no new clauses:
            #   x: aa+=1, pc stayed same
            #   y: aa+=1, pc-=1
            # print(f"rx {laa} {lpc}  a {delta_from_prev_state.len_new_availableActions} {len(delta_from_prev_state.removed_availableActions)} " +
            #        f"p {delta_from_prev_state.len_new_processed_clauses} {len(delta_from_prev_state.removed_processed_clauses)}" +
            #       ("" if gopts().deterministic_randseed
            #       # f"{('y' if laa==olaa-1 and lpc==olpc else 'x' if laa==olaa-1 and lpc==olpc+1 else 'z') if len(newunprocessed_clauses_strs) == 0 and len(newprocessed_clauses_strs)==0 else ' '} "
            #       # f"{tm - tm0:8.6f} {tm:8.4f} {eptime}")
            #       else f" {tm - tm0:8.6f} {tm:8.4f}"))

            return selected_clause, prover_delta, next_state_processed_clauses_set, next_state_availableActions_set, \
                 False, False, set([])

        else:
            final_resolvent, no_proof_found, used_in_proof_list = self.read_final_eprover_output()
            return selected_clause, None, set([]), set([]), \
                 final_resolvent, no_proof_found, used_in_proof_list

    def read_final_eprover_output(self):
            if "CHECK_ALPHA" in os.environ:
                print('ALPHA',len(self.all_clauses), len([tuple(sorted(str(x))) for x in self.all_clauses]))
            print(f"Proof found? {len(self.all_clauses)}")
#             os.system(f"sed -n '/# Proof found!/,$p' {dfnames().eprover_out} > eprover.proof")
            final_resolvent = (os.system(f"grep -q '# Proof found!' {dfnames().eprover_out}") == 0)
            no_proof_found = 0 if final_resolvent else (os.system(f"grep -q '# No proof found!' {dfnames().eprover_out}") == 0)

            # clause_lens={}
            # for clause in self.all_clauses:
            #     ln = len(str(clause))
            #     # print(f'xlen {ln} {clause} {str(clause)}')
            #     if ln in clause_lens:
            #         clause_lens[ln] += 1
            #     else:
            #         clause_lens[ln] = 1
            # for ln,n in clause_lens.items():
            #     print('len',ln,n)

            if no_proof_found:
                Path('eprover.noproof').touch() # used in launch-ee.sh; also single_player_coach.py. yuck.
            if not final_resolvent and not no_proof_found:
                print(f"Proof NOT found!")
                return False, False, set([])
            else:
                print(f"Proof found!" if final_resolvent else "No proof found!")
                trainpos_file = "eprover.trainpos"
                # delete everything up to the start pattern; then everything after the end pattern
                if final_resolvent:
                    os.system(f"sed '1,/# Training: Positive examples begin/d; /# Training: Positive examples end/,$d' eprover.txt > {trainpos_file}")
                else:
                    # the launch scripts look for this file
                    os.system(f"sed '1,/# SZS output start Saturation/d;       /# SZS output end Saturation/,$d'       eprover.txt > eprover.noproof")
                    return False, True, [] # I suspect that this set may be empty, so need the bool


                return True, False, self.read_trainpos(trainpos_file, no_proof_found) # I suspect that this set may be empty, so need the bool

    def read_trainpos(self,trainpos_file,no_proof_found=False):
        selected_clauses_map = {}
        if os.environ["CHECK_SELECTED_CLAUSE"]:
            with open("selected_clause.txt", "r") as f:
                for idx, line in enumerate(f.readlines()):
                    # print('linex',line)
                    transformed_cl, _, _ = self.parser.parse(line)
                    # print('linex',transformed_cl)
                    selected_clauses_map[transformed_cl[0]] = self.selected_clauses[idx]

        suffix = "# trainpos\n"
        clauses_used_in_proof = []
        with open(trainpos_file, "r") as f:
            for line in f.readlines():
                line = line[0:len(line) - len(suffix)]
                if no_proof_found and line.startswith("fof("):
                    print(f"Skipping fof no-proof line: {line}")
                    continue
                eprover_id, formula = EProver.split_fof(line)
                #                         _, transformed_cl = self.fof_to_clause_with_caching(line)
                transformed_cl, _, _ = self.parser.parse(formula)
                if len(transformed_cl) > 0:
                    clause=transformed_cl[0]
                    if selected_clauses_map:
                        if clause in selected_clauses_map:
                            if clause != selected_clauses_map[clause]:
                                print(f"substituting: {clause} => {selected_clauses_map[clause]}")
                                clause = selected_clauses_map[clause]
                        else:
                            print('clause not in selected_clauses_map!', clause)
                    clauses_used_in_proof.append(clause)
                else:
                    raise Exception(f"Failed to parse positive clause: {line}")
        return clauses_used_in_proof

    # this comment is out of date
    # We modified E so that at each iteration it first prints out any clauses that are 'new'.
    # This is what I'm calling 'registering'; we read in the clauses and implicitly assign them
    # the same index that the new code in E does.
    # This index is NOT necessarily the same as E's internal id, because sometimes after a clause
    # is removed, it reappears due to simplification; in that case, we assign it a new id,
    # just so that we have the nice property that every clause in this list has a definite lifetime;
    # that is, we can say exactly when the clause is removed.
    # In the case of duplicates, each will have its own removal iteration.
    def register_new_clauses(self, new_clause_strs, selected_literals, discard):
        start_time = time.time()
        for idx, cl in enumerate(new_clause_strs):
            if discard:
                print(idx, file=self.trailOutput, flush=True)
            else:
                cl = cl.strip()
                eprover_id, formula = EProver.split_fof(cl)

    #             assert eprover_id.startswith("i_0_") # this is not true in the proof/trainpos fofs
    #             epid = int(eprover_id[4:])

                if formula.count('(') != formula.count(')'):
                    # this is a common result if the output is truncated because eprover quits
                    print('imbalanced parens',idx, len(new_clause_strs), formula.count('('), formula.count(')'), cl)
                if selected_literals:
                    transformed_cl, sellit,sorted_lits = self.parser.parse(formula, selected_literals, idx)
                else:
                    transformed_cl, sellit,sorted_lits = self.parser.parse(formula)
                assert len(transformed_cl)==1
                clause = transformed_cl[0]
                # it may be that eprover sorts the lits, so formula changes but clause is the same
                # the sellit need not be updated, since (presumably) the same lits are chosen
                if not clause in self.clause2id:
                    # print('register0', formula)
                    # print('register1', clause)
                    if formula.count('!=') != str(clause).count('!='):
                        print('!= mismatch!')
                    clauseid = len(self.all_clauses)
                    # if "RV" in os.environ:
                    #     print('register1', clauseid,clause)
                    self.clause2id[clause] = clauseid
                    self.episode.age[clause] = self.step
                    self.all_cnf.append(cl) # for dumping later
        #             print('register', len(self.all_clauses), cl)
                    self.all_clauses.append(clause)
                    self.all_clauses_actions.append((clause, FullGeneratingInferenceSequence))
                    self.episode.selected_literal[clause] = sellit
                    if self.clause_hasher != None:
                        self.clause_hasher.set_clause_hash(clause, str(sorted_lits))
                    # if sellit:
                    #     print('reg', sellit, clause.literals[sellit], clause)
                    # elif len(clause.literals)>1:
                    #     print('reg', sellit, clause)
                    # else:
                    #     print('reg', clause)
                    # self.all_clauses_actions.append((clauseid, FullGeneratingInferenceSequence))
                    # self.all_clauses_preds[clause] = set([literal.atom.predicate for literal in clause.literals])
                    # print('nx', self.all_clauses_preds[clause], [type(literal.atom.predicate) == EqualityPredicate for literal in clause.literals],clause)
                    if cl.split(",", 3)[1].strip() == "negated_conjecture":
                        self.episode._derived_from_neg_conj.add(clause)
                print(self.clause2id[clause], file=self.trailOutput, flush=True)
        self.trailOutput.flush()
        EProver.register_new_clauses_time += time.time() - start_time

    # All clauses have been registered, we just need to look them up.
    def fill_processed_clauses(self, trailids):
        # return trailids
        return [self.all_clauses[ind] for ind in trailids]
            
    def fill_available_actions(self, actions):
        clauses_processed = self.fill_processed_clauses(actions)
        self.eprover_available_actions = clauses_processed # this is in the order that eprover sends, NOT the re-ordered one
        return [self.all_clauses_actions[ind] for ind in actions]

    # def derived_from_negated_conjecture(self):
    #     return self._derived_from_neg_conj

    # @timeout_decorator.timeout(60)
    def getStateStrings(self,discard=False,printstrs=False):
        start_time = time.time()
        def readints(f,tp):
                x = array.array(tp)
                x.fromfile(f, 1)
                n=x.pop()
                # print(f'ri {n}', flush=True)
                x.fromfile(f, n)
                return x

        def readstrs(f):
            strs = []
            # print('readstrs0', flush=True)
            s = f.readline()
            nonlocal printstrs
            if printstrs:
                print('readstrs', s.strip(), flush=True)
            while s != '\n':
                if not s:
                    break
                if not s.endswith('\n'):
                    print('NO NEWLINE IN STRING: presumably eprover died', s)
                    exit(1)
                # print('nupx', s)
                strs.append(s.strip())
                s = f.readline()
                if printstrs:
                    print('readstrs', s.strip(), flush=True)
            # print('readstrsdone', flush=True)
            return strs

        # print('wp', flush=True)
        try:
            newcls = readstrs(self.trailInput)
            # print('wp2', flush=True)
            selected_literals=None
            if os.environ["TE_GET_LITERAL_SELECTION"]:
                # print('wp21', flush=True)
                selected_literals = readints(self.trailInputBin,'L')
                assert len(newcls)<=len(selected_literals) # this now contains starts for each clause and after that the values
            # we need both the clause and its selected lit to register it
            self.register_new_clauses(newcls,selected_literals,discard)

            unprocessed_clauses_strs = readints(self.trailInputBin,'i')

            # print('wp3', flush=True)
            processed_clauses_strs = readints(self.trailInputBin,'i')
            # print('wp4', flush=True)
            # print('aaids', unprocessed_clauses_strs, flush=True)
            # print('pcids', processed_clauses_strs, flush=True)
        except EOFError:
            if (os.system("grep -q 'eprover: CPU time limit exceeded, terminating") == 0):
                print('ignoring EOFErr, eprover ran out of time')
                return [],[]
            else:
                raise

        EProver.receive_state_time += time.time() - start_time
        # for f in ["newunprocessed", "unprocessed", "newprocessed", "processed"]:
        #     os.unlink(f)
        return processed_clauses_strs, unprocessed_clauses_strs

    @staticmethod
    def split_fof(clause):
        #print(f"\toriginal clause: {clause}")
        start_method = time.time()
        clause = clause.strip()
        eprover_id = None
        ret = None
        if len(clause)>0 and not clause.startswith("cnf("):
            eprover_id = clause
            formula = ""
        elif len(clause)>0:
            if clause[-1] == ".":
                formula = clause[:-1]
            else:
                formula = clause
            split = formula.split(",",2)
            eprover_id = split[0].split("(")[1].strip()
            formula = split[2][:-1].strip()
        else:
            formula =""
        return eprover_id, formula

    prover_generation_time = 0
    prover_retention_time = 0
    prover_simplification_time = 0
    prover_action_selection_time = 0
    prover_total_time = 0
    prover_clone_time = 0
    @staticmethod
    def print_prover_runtime_stats():
        if True:
            # these fields weren't actually updated for EProver
#             print(("Reasoner total time {} secs: {} generation ({} %), " +
#                    "{} retention ({} %), {} simplification ({} %), {} action section ({} %)")
#                   .format(ActiveState.prover_total_time, ActiveState.prover_generation_time,
#                           int(100 * ActiveState.prover_generation_time / ActiveState.prover_total_time),
#                           ActiveState.prover_retention_time, int(100 * ActiveState.prover_retention_time / ActiveState.prover_total_time),
#                           ActiveState.prover_simplification_time,
#                           int(100 * ActiveState.prover_simplification_time / ActiveState.prover_total_time),
#                           ActiveState.prover_action_selection_time,
#                           int(100 * ActiveState.prover_action_selection_time / ActiveState.prover_total_time)))
            print(f"\tE Prover register new clauses time: {EProver.register_new_clauses_time}")
#             print(f"\tE Prover actual parsing time (without caching): {EProver.parse_without_cache}")
#             print(f"\tE Prover actual parsing time (with caching): {EProver.parse_with_cache}")
#             print(f"\tE Prover core parsing time (without caching): {ParsingTime.core_parsing_time}")
            print(f"\tE Prover time to receive state: {EProver.receive_state_time}")
#             print(f"\tE Prover state string transform time: {EProver.state_string_processing_time}")

#         print("Prover clone time {} secs".format(ActiveState.prover_clone_time))
#         print(f"State creation and maintainance time {ActiveState.state_creation_update} secs")


