from logicclasses import Clause
from typing import Set
class ClausePool:
    def __init__(self):
        self.pool = {}
        self.reuse_count = 0

    def get(self, clause:Clause):
        ret = self.pool.get(clause, None)
        if ret is None:
            self.pool[clause] = clause
            ret = clause
        elif clause is not ret:
            self.reuse_count += 1
        return ret

    def retain(self, clauses:Set[Clause]):
        to_remove = set([])
        for key in self.pool.keys():
            if key not in clauses:
                to_remove.add(key)

        for key in to_remove:
            del self.pool[key]
    def size(self):
        return len(self.pool)
