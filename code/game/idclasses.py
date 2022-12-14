import re
import dataclasses

@dataclasses.dataclass(frozen=True, eq=True)
class ClauseTID:  # Training Id
    iteration: int
    training: bool
    episode_num: int
    index: int

@dataclasses.dataclass(frozen=True, eq=True)
class ProblemID:
    ds_prefix: str
    problem_num: int

    def __str__(self) -> str:
        return f"{self.ds_prefix}{self.problem_num}"

def parseProblemId(episode_str:str)->str:
    episode_ds = episode_str[0:re.search('\d',episode_str).start()]
    episode_num = int(episode_str[len(episode_ds):])
    return ProblemID(episode_ds,episode_num)

