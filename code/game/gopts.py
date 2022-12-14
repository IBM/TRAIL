import yaml
import dataclasses
import sys, os, numpy, random, torch
from typing import List, Tuple, Dict, Any, Optional
from dfnames import dfnames

__all_ = ['gopts', 'setGOpts']

@dataclasses.dataclass(frozen=True, eq=False)
class GOpts:
    """Singleton class to contain global, hopefully immutable options.  See optdefaults.yaml for documentation"""

    # To add a field, just:
    # 1) add the field here, and
    # 2) add a default entry in optdefaults.tcfg (at the end of the file, not the start where the bash vars are)
    # The program doesn't print doc strings; put documentation in optdefaults.tcfg.
    # the order of fields in this file is unimportant; there is no relation to the ordering in optdefaults.
    # In a very few cases there are settings that concern both eprover and the python code;
    # in those cases we set an env var and access its value in python using os.environ;
    # I don't bother add it to this class.

    max_actions: int
    train_stopping_factor: int
    train_min_number_of_steps : int

    treat_constant_as_function: bool
    
    include_action_type: bool
    
    clause_embedding_layers: int
    clause_embedding_size: int

    clause_feat_aggr: str
    
    numItersForTrainExamplesHistory: int
    value_loss_weight: float
    
    append_age_features: bool
    only_age_features: bool
    
    nnet_model: str
    
    # num_attn_heads: int
    # heads_max: bool
    
    optimizer: str
    
    epochs: int
    epochs_iter1: int

    compute_temp: str

    grad_dot_prod: bool
    
    entropy_reg_loss_weight: float

    advantage: bool
        
    keep_emb_size: int
    filter_collected_examples: bool
    num_gcn_conv_layers: int
    num_node_embedding_linear_layers: int
    node_char_embedding_size: int
    embedding_activation: str
    use_sparse_gcn_embed: bool
    mpnn_node_state_dim: int
    mpnn_edge_state_dim: int
    mpnn_iters: int
    mpnn_node_ct: int
    mpnn_edge_ct: int
    random_predict: bool
    value_net_layers: int
    beta1: float
    beta2: float
    proc_feat_type: str # {convex_all, convex_conj, conj_only, simple_sum, combined_only, weighted_all, weighted_conj, simple_gate_conj, simple_gate_all}
    early_stopping_patience: int
    weight_reward_pred_useless: float

    policy_loss_weight: float
    reward_breadth_weight: float
    use_useful_fraction: bool
    value_net_units: int
    graph_embedding_output_size: int

    graph_embedding_add_self_loops: bool
    max_reward_per_problem: float
    num_syms: int
    reward_depth_weight: float
    
    keep_top_attempts_per_problem: int
    wd: float
    use_time_for_reward_calc:bool
    # graph_embedding_output_size: int
    enigma_size: int
    enigma_seq_len: int
    incl_enigma_subseq: bool
    predicate_term_anonymity_level: int
    # graph_embedding_add_self_loops: bool

    max_score: float
    binary_reward: bool
    discounted_problems: List[str]
    discounted_reward_drop_threshold: float
    early_stopping_thresold: float

    eval_interval: int
    herbrand_vector_size: int
    init_state_name_node_embeddings: bool
    lr: float
    min_reward_per_problem: float
    problem_discount_factor: float
    reward_norm: int

    batch_size: int
    cuda: bool
    discount_factor: float # really: None or float, so handled specially during validation

    dropout_p: float
    penalty: float
    pos_example_fraction: float

    conj_aggr_type: str
    incl_subchains: bool
    max_age: int
    max_literal_ct: int
    max_weight_ct: int
    proc_nc_comb_layers: int
    sos_feat: bool
    vectorizer_arch: str
             
    vectorizer: str

    charcnn_filters: str 
    embedding_use_bias: bool
    embedding_use_cnn_gate: bool
    graph_embedding_add_back_edges: bool
    heterogeneous_edges: bool
    node_type_embedding_size: int
    use_one_hot_action_types: bool
    use_one_hot_node_types: bool
    
    print_caching_stats: bool
    
    first_epoch_saved_model: int
    graph_embedding_max_depth: int
    gcn_skip_connections: bool
    root_readout_only: bool
    
    real_time_limit: bool
    ignore_initialization_time: bool

    ez_reward: bool
    reward_all_choices: bool
    only_new_examples: bool

    state_next_step: bool
    pool_clauses: bool
    save_id2clausemap: bool
    use_clause_pool: bool
    # use_pickle_for_clause_pool: bool
    clause_2_graph: bool
    # shuffle_examples: bool
    # examples_by_episode: bool
    # cache_by_episode: bool
    no_gpu_gz_dir: bool
    ignore_negation: bool
    ignore_predicate_names: bool
    only_use_best_episodes: bool

    deterministic_randseed: int

    use_HER: bool
    step_limit: int
    fill_removed_positions: bool
    drop_index: bool
    use_InitStep: bool

    print_clause_hashes: bool

    # even after setting PYTHONHASHSEED this is apparently necessary
    def maybe_sort_list(self,x):
        if self.deterministic_randseed:
            return sorted(x, key=lambda x: str(x))
        return list(x)

    def maybe_sort_set(self,x):
        if self.deterministic_randseed:
            return sorted(x, key=lambda x: str(x))
        return x

    def __post_init__(self):
       #https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
#         for opts in ['trainopts']:
#             object.__setattr__(self, opts, ConfigOpts(**getattr(self, opts)))
              
       validate(self)

# @dataclasses.dataclass(frozen=True, eq=False)
# class ConfigOpts:
#     data_dir: str
#     problems_dir: str
#     axioms_dir: str 
#     
#     def __post_init__(self):
#        validate(self)
# 
# 
# def modifyConfigDefaults(dict, trail_dir):
#     dict.data_dir = dict.get('data_dir', trail_dir+"/train_data").replace("$HOME", os.environ.get('HOME'))
# #     print('trail_dir:', trail_dir)
#     dict.problems_dir = dict.get('problems_dir', dict.data_dir+"/Problems").replace("$HOME", os.environ.get('HOME'))
#     
#     dict.axioms_dir = dict.get('axioms_dir', dict.data_dir+"/Axioms").replace("$HOME", os.environ.get('HOME'))
#     
#     dict.cached_parses_dir = dict.get('cached_parses_dir', dict.data_dir+"/CachedParses").replace("$HOME", os.environ.get('HOME'))
# 
#     return dict

_goptsy_: Optional[GOpts] =None
goptsx: Optional[Dict[str,Any]]   =None

def modifyDefaults(dict, cuda:bool, showdiffs:bool) -> Dict[str,Any]:
    optdefaults_file = "expdir/optdefaults.yaml"
    print(f"loading yaml defaults from {optdefaults_file}")
    with open(optdefaults_file, "r") as f:
        optdefaults = yaml.load(f.read(), yaml.SafeLoader)
        # print(yaml.dump(optdefaults))
        # print('dumped')
#     print(optdefaults)
#     print(type(optdefaults))
#     print(optdefaults.items())
#     if "numEps" in optdefaults:
#         print('gotit')
#     print(type(numEps), numEps)
#     assert optdefaults.numEps == 0 # causes error - why?
#         optdefaults.update({'numEps':numEps})
#         print(optdefaults.items())

#     optdefaults.numEps = numEps # this doesn't work - why?

#     if not('numEps' in dict and dict.numEps>0):
#     if dict.get('numEps',0)==0:
#         print('setting')
#         dict.update({'numEps':numEps})
#     print('set numEps to ', dict.get('numEps'))

    if dict.get('cuda'):
        sys.exit("Don't actually set the 'cuda' field, it is done automatically.")
    dict["cuda"] = cuda
    if showdiffs:
        print("\nnon-default options:")
    for key,defval in optdefaults.items():
#         if dict.get(key) == None:
            
        dval = dict.get(key)
        #print('dict',key,dval,type(dval))
        if key == 'discount_factor':
            # complete hack, until I learn how to represent None in yaml
#             assert dval == "None", dval

            # from old main.py:
#     if args.discount_factor == 1.0:
#         print(f"WARNING: For backward compatibility, a discount factor of 1.0 is interpreted as disabling discounted reward")
#         args['discount_factor'] = None

            if dval == 1.0:
                sys.exit(f"WARNING: For backward compatibility, a discount factor of 1.0 is interpreted as disabling discounted reward - CHANGE YOUR INPUT to None")
            elif dval and dval!=12345678.0:
                if showdiffs:
                    print(f"{key}: {dval}")
                print('discount_factor is now ', dval)                
            else:
                dict.update({key:None})
                print('discount_factor is now ', None)
        elif dval == None:
            dict.update({key:defval})
#         elif type(defval) == dict:
#             # only one dict...
#             dict.update({key:modifyConfigDefaults(dval, trail_dir)})
#         elif dval != defval:
#             if type(defval) == str:
#                 dval = dval.replace("$HOME", os.environ.get('HOME'))
#             if showdiffs:
#                 print(f"{key}: {dval}")
        else:
            # explicity passed the default
            pass 
    if showdiffs:        
        print("end non-default options\n")
    
#     os.environ.get('HOME')
    return dict

def gopts() -> GOpts:
    global _goptsy_
    assert _goptsy_
    return _goptsy_

def setGOpts(yamlfile:str, cuda:bool, showdiffs=False):
    try:
#         experiment_dir = os.getcwd()
        global goptsx, _goptsy_
        if "BOTTLENECK" in os.environ and _goptsy_ != None:
            return
        assert _goptsy_ == None # only call once
#         yamlfile = experiment_dir + "/opts.yaml"

#         numEps = len(os.listdir(os.path.join(experiment_dir, "episodes")))
        if yamlfile:
            print(f"loading yaml from {yamlfile}")        
            with open(yamlfile, "r") as f:
                document = f.read()
            goptsx = yaml.load(document, yaml.SafeLoader)
            if isinstance(goptsx,str):
                print("This doesn't seem to be a valid yaml file:",goptsx, file=sys.stderr)
                sys.exit(1)

            if not goptsx:
                print('This yaml file is apparently empty')
                goptsx={}
#             print(f"loaded yaml file {goptsx}")
        else:
            print("No yaml options file specified - using all defaults")
#             with open(dfnames().yamlopts0_file, "r") as f:
#                 document = f.read()
#             goptsx = yaml.load(document, yaml.SafeLoader)            
            goptsx = dict()
#             goptsx['model0'] = False

        assert goptsx != None
        goptsx = modifyDefaults(goptsx, cuda, showdiffs)
        assert goptsx != None
        _goptsy_ = GOpts(**goptsx)
        numpy.seterr(all='print')  # print to stdout
        seed = _goptsy_.deterministic_randseed
        #if not seed.is_digit:
        # print('deterministic_randseed',seed,type(seed),isinstance(seed, int), type(seed)==type(1))
        # if not isinstance(seed, int):
        if type(seed) != type(1):
            sys.exit(f"deterministic_randseed must be an int: {seed}")

        # COMMENT OUT THIS SECTION IF YOU STILL HAVE PROBLEMS WITH TIME_LIMIT
        try:
            tls = os.environ["TIME_LIMIT"]
            print(f"TIME_LIMIT={tls} (in gopts.py)")
            tl=int(tls)
        except:
            sys.exit(f"env var TIME_LIMIT must be an int") # eprover will fail if this is a float
        # END OF SECTION TO COMMENT OUT

        if seed:
            msgs=[]
            if "PYTHONHASHSEED" not in os.environ:
                msgs.append(f"It only makes sense to set deterministic_randseed if you've also set PYTHONHASHSEED to some (non-0) value")
            # https://pytorch.org/docs/stable/notes/randomness.html
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            if not _goptsy_.step_limit:
                msgs.append("You must set step_limit to non-0 for deterministic_randseed")
            if tl:
                msgs.append(f"You must set TIME_LIMIT to 0 for deterministic_randseed ({tl})")
            if msgs:
                for s in msgs: print(s)
                sys.exit(1)
            assert torch.random.initial_seed() == seed
            # https://discuss.pytorch.org/t/use-deterministic-algorithms/118600
            if torch.__version__.startswith('1.7'):
                torch.set_deterministic(True)
            else:
                assert 0 # mypy doesn't like this; if we ever use <1.7, have to use this method
                # torch.use_deterministic_algorithms(True)



    except FileNotFoundError as e:
        print(e) 
        sys.exit(f"couldn't open yaml file {yamlfile}")

#     except TypeError as e:
#         print(e) 
#         sys.exit(">>>> The yaml file didn't parse")

    except yaml.YAMLError as e:
        print(e) 
        sys.exit(">>>> The yaml file didn't parse")
    except ValueError as e:
        print(e) 
        sys.exit(">>>> A global option doesn't have the right type")
        

#https://stackoverflow.com/questions/51736938/how-to-validate-typing-attributes-in-python-3-7
def validate(instance):
    for field in dataclasses.fields(instance):
        attr = getattr(instance, field.name)
        if field.name == 'discounted_problems': # HACK, because we can't test for List[str]
#             print('e_options type', attr)
            if not isinstance(attr, list):
                raise ValueError(f"Field {field.name} is not of type List")
            for s in attr:
                if not isinstance(s, str):
                    raise ValueError(f"member of Field {field.name} is not of type str")
        elif field.name == 'discount_factor':
            if not (attr is None or isinstance(attr, float)):
                raise ValueError(f"field {field.name} is not None or of type float: {attr} {type(attr)}")                
        elif not isinstance(attr, field.type):
            raise ValueError(f"Field {field.name} is of type {type(attr)}, should be {field.type}")

    if not instance.advantage and instance.value_loss_weight != 0.0:
        sys.exit(f"If 'advantage' is False, then we require value_loss_weight==0.0, but you set it to {instance.value_loss_weight}\n"+
                 "fix that and try again")
    if instance.binary_reward:
        if instance.max_score:
#            'max_score': 0 if int(parsed_gopts().binary_reward) else float(parsed_gopts().max_score),            
            sys.exit('since binary_reward is set, then max_score must be 0')
 
    args = instance

    assert args.pos_example_fraction >= 0
    assert args.pos_example_fraction <= 1
    assert args.discounted_reward_drop_threshold is None or args.discounted_reward_drop_threshold < 1

    
    # if instance.num_attn_heads != 1:
    #     sys.exit("The code has not been tested with num_attn_heads!=1!  You can remove this error, but be warned.")
        
    if instance.discounted_problems:
        sys.exit("The option 'discounted_problems' doesn't actually work!  Sorry.")
    # print('validate', instance)

if __name__ == "__main__":
    setGOpts(dfnames().yamlopts_file, False)
    for field in dataclasses.fields(_goptsy_):
        attr = getattr(_goptsy_, field.name)
        print(f"{field.name}: {attr}")
    print('option settings are valid')

    # this is one of the few times  I trust the exit code, since this is very small.
    # trail.sh will fail if this returns non-0.
    sys.exit(0)