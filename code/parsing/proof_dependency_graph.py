import sys, traceback, re

def get_step_diff(step_deps, all_axioms_from_proof, steps_dic):
    arr = step_deps.split(',')
    diff = -1
    if set(arr).issubset(all_axioms_from_proof):
        diff = 1
    else:
        i = 2
        sub_arr_not_in_axioms = []
        for a in arr:
            if a not in all_axioms_from_proof:
                sub_arr_not_in_axioms.append(a)
        assert len(sub_arr_not_in_axioms) > 0
        filtered_sub_arr_not_found_yet = sub_arr_not_in_axioms.copy()
        for key, value in steps_dic.items():
            for a in sub_arr_not_in_axioms:
                if a in value:
                    filtered_sub_arr_not_found_yet.remove(a)

            # if set(sub_arr_not_in_axioms).issubset(value):
            #     diff = i
            #     break
            diff = i
            if len(filtered_sub_arr_not_found_yet) == 0:
                break
            i += 1
    return diff

def get_step_dependencies(step_str):
    # match = re.search(r"\[([A-Za-z0-9_,]+)\]", step_str)
    # if match and len(match.groups()) > 0:
    #     return match.groups()[len(match.groups())-1]

    matches = re.findall(r"\[([A-Za-z0-9_, ]+)\]", step_str)
    if matches and len(matches) > 0:
        return matches[len(matches)-1]
    else:
        return None
