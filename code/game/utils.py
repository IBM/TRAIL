import sys
import hashlib

class ClauseHasher:
    def __init__(self):
        self.clause_hash = {}
        self.clause_str = {}
        self.selected_clauses = []

    def set_clause_hash(self, clause, s:str):
        self.clause_str[clause] = s
        self.clause_hash[clause] = hashlib.shake_128(s.encode('utf-8')).hexdigest(16)

    def hash_clause(self, clause):
        return self.clause_hash[clause]

    def hash_clauses(self, extra): # yuck, have to share with eprover.py
        m = hashlib.shake_128()
        if extra:
            clauses = self.selected_clauses + [extra]
        else:
            clauses = self.selected_clauses
        for x in sorted([self.clause_hash[clause] for clause in clauses]):
            m.update(x.encode('utf-8'))
        print('hash_clauses',m.hexdigest(16))
        return m.hexdigest(16)

class dotdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError


#https://stackabuse.com/handling-unix-signals-in-python/
def receiveSignal(signalNumber, frame):
    print('Received signal:', signalNumber)
    sys.exit(0)

#     signal.signal(signal.SIGPIPE, signal.SIG_IGN)
#     signal.signal(signal.SIGPIPE, receiveSignal)
    
def dumpObjBasic(msg, obj):
    dumpDictBasic(msg, vars(obj))
    
def dumpDictBasic(msg, dict):
    print('dict ', msg)
    for key,val in dict.items():
        tp = type(val) 
        if tp==int or tp==str or tp==float:
            print(key, ':', val)
        elif val==None:
            print(key, ': None')
        else:
            print(key, ': ...');
       
def printToFile(fname, x):
    with open(fname, 'w') as f:     
        f.write(x)