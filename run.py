"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Entry point for the creation of the variable elimination algorithm in Python 3.
Code to read in Bayesian Networks has been provided. We assume you have installed the pandas package.

"""
from read_bayesnet import BayesNet
from variable_elim import VariableElimination
from em_algorithm import EMAlgorithm, load_data
import copy

if __name__ == '__main__':
    # The class BayesNet represents a Bayesian network from a .bif file in several variables
    net = BayesNet('earthquake.bif') # Format and other networks can be found on http://www.bnlearn.com/bnrepository/
    
    # These are the variables read from the network that should be used for variable elimination
    print("Nodes:")
    print(net.nodes) 
    print("Values:")
    print(net.values)
    print("Parents:")
    print(net.parents)
    print("Probabilities:")
    print(net.probabilities)

    # Make your variable elimination code in the seperate file: 'variable_elim'. 
    # You use this file as follows:
    ve = VariableElimination(net)

    # Set the node to be queried as follows:
    query = 'Alarm'

    # The evidence is represented in the following way (can also be empty when there is no evidence): 
    evidence = {'Burglary': 'True'}

    #MAP variables
    map_variables = ['JohnCalls', 'Earthquake']

    # Remove variables opposite of those observed (observed 'Burglary': True, so we remove 'Burglary': False)
    cpts = [df for df in net.probabilities.values()]
    updated_cpts = []
    for df in cpts:
        for key, value in evidence.items():
            if key in df.columns:
                df = df[df[key]==value]
        updated_cpts.append(df)

    # Determine your elimination ordering before you call the run function. The elimination ordering   
    # is either specified by a list or a heuristic function that determines the elimination ordering
    # given the network. Experimentation with different heuristics will earn bonus points. The elimination
    # ordering can for example be set as follows:
    def minimum_factor_size(factor, evidence=set()):
        if evidence is None:
            return {}
        product_formula = [list(df.columns[df.columns != 'prob']) for df in factor]
        elim_order = []
        variables = {var for factor in product_formula for var in factor} - set(evidence.keys()) #All variables - evidence variable(s)

        while variables:
            sizes = {}
            for var in variables:
                involved = [set(factor) for factor in product_formula if var in factor]
                union_involved = set().union(*involved) #Concatenate all variables in involved into one set, removing duplicates and returning 1 single set.
                sizes[var] = 2**len(union_involved) #2^k for boolean operations. 

            best_var = min(sizes, key=sizes.get)
            elim_order.append(best_var)
            variables.remove(best_var)

        return elim_order

    #####################################################################################################################
    #Variable Elimination:

    # Call the variable elimination function for the queried node given the evidence and the elimination ordering as follows:
    with open("variable_elimination.log", "w") as log_file:   
        result_ve = ve.run(query, updated_cpts, minimum_factor_size(updated_cpts, evidence), evidence, log=log_file) #Set to None if no logging is wanted
    print(f"Result for query variable(s) {query} with evidence {evidence} and elimination order {minimum_factor_size(updated_cpts, evidence)}:")
    print(result_ve)

    #####################################################################################################################
    #MAP:

    #Exclude MAP variables from evidence
    exclude = dict(evidence)
    for element in map_variables:
        exclude[element] = ""
    elim_order_map = minimum_factor_size(updated_cpts, exclude)

    #Call Variable Elimination with MAP variables algorithm:
    with open("map.log", "w") as log_file:
        result_map = ve.run_with_map(map_variables, updated_cpts, elim_order_map, evidence, log=log_file)
    print(f"Result for MAP variable(s) {map_variables} with evidence {evidence} and elimination order {elim_order_map}:")
    print(result_map)

    #####################################################################################################################
    #EM learning algorithm:
    net_original = BayesNet('endorisk_new.bif')
    data = load_data('simulation_data_hid_names.dat', net_original)
    data = data.head(1000)

    best_network = None
    best_loglikelihood = float('-inf')

    for i in range(5):
        net_copy = copy.deepcopy(net_original)
        with open(f'em_run_{i+1}.log', 'w') as log_file:
            em = EMAlgorithm(net_copy, data, log=log_file)
            learned, loglikelihood = em.run(max_iter=50)
        print(f"Run {i+1}: final log-likelihood = {loglikelihood:.4f}")
        if loglikelihood > best_loglikelihood:
            best_loglikelihood = loglikelihood
            best_network = learned

    #Display best log-likelihood:
    print(f"Best log-likelihood = {best_loglikelihood:.4f}")
    #####################################################################################################################