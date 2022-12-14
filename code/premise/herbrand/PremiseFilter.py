import torch
from premise.herbrand.MLP import MLP
# from premise.herbrand.PremiseDataset import create_vectorizer
from game.vectorizers import HerbrandVectorizer

def create_vectorizer(all_problem_clauses):
    return HerbrandVectorizer(all_problem_clauses, vectorizer = 'mem_htemplate',
                              d=None, max_ct=500,
                              num_symmetries=0, herbrand_vector_size = 550, append_age_features=False)


class PremiseFilter:
    args = None
    model = None
    device = None
    @staticmethod
    def initalize(args, premise_model_path):
        '''
        :param config: a dictionary with configuration parameters
        :param max_actions: the maximum number of actions that can be performed at a given
        :param repeat play a particular problem repeat times before moving to the next
        decision point
        '''
        PremiseFilter.args = args
        print('initalize PremiseFilter.args: ', PremiseFilter.args)

        if PremiseFilter.model is None:
            print('Trying to load premise model: ', premise_model_path)

            # input_size = (args.herbrand_vector_size) * 2
            input_size = 550 * 2
            PremiseFilter.model = MLP(input_size, hidden_size=input_size * 2, output_size=1)
            if torch.cuda.is_available():
                PremiseFilter.device = torch.cuda.current_device()
            else:
                PremiseFilter.device = torch.device('cpu')
                # device = 'cpu'
            if torch.cuda.is_available():
                PremiseFilter.model.load_state_dict(torch.load(premise_model_path))
            else:
                PremiseFilter.model.load_state_dict(torch.load(premise_model_path, map_location=torch.device('cpu')))
            print('Premise model: ', premise_model_path, ' is loaded successfully!')
        else:
            print('Model is already loaded!!')
    @staticmethod
    def filter_problem_premises(negated_conjectures, clauses):
        if type(clauses) != list:
            clauses = list(clauses)
            negated_conjectures = list(negated_conjectures)
        print('PremiseFilter.args: ', PremiseFilter.args)
        conj_clauses_vec = []
        ax_clauses_vec = []
        vectorizer = create_vectorizer(negated_conjectures[0])
        for clause in negated_conjectures[0]:
            conj_clauses_vec.append(vectorizer.clause_vectorization(clause, ''))

        vectorizer = create_vectorizer(clauses)
        for clause in clauses:
            ax_clauses_vec.append(vectorizer.clause_vectorization(clause, ''))

        examples = []
        # print('type(conj_clauses_vec): ', type(conj_clauses_vec))
        # print('type(clauses): ', type(clauses))

        print('Number of caluses in conjecture: ', len(conj_clauses_vec), ', Number of axiom clauses: ', len(clauses))
        conj_vec_sum = None
        for conj in conj_clauses_vec:
            if conj_vec_sum is None:
                conj_vec_sum = torch.tensor(conj)
            else:
                conj_vec_sum += torch.tensor(conj)

        for pos in ax_clauses_vec:
            prem_tensor = torch.tensor(pos)
            examples.append(torch.cat((conj_vec_sum, prem_tensor)).float())

        # torch_list = torch.cat(examples, dim=2)
        torch_list = torch.stack(examples).to(PremiseFilter.device)

        outputs = PremiseFilter.model(torch_list)  # torch.Size([B, 1])
        outputs = torch.sigmoid(outputs)
        predicted = torch.round(outputs)
        print(predicted)
        print(predicted.shape)
        pos_axs = []
        for i, val in enumerate(predicted):
            if val == 1:
                pos_axs.append(clauses[i])
        print(f'Predicted positive axioms = {len(pos_axs)}, out of {len(clauses)} ')
        return pos_axs