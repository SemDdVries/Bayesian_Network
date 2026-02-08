"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Entry point for the creation of the variable elimination algorithm in Python 3.
Code to read in Bayesian Networks has been provided. We assume you have installed the pandas package.

"""
from read_bayesnet import BayesNet
from variable_elim import VariableElimination

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

    # Remove variables opposite of those observed (observed 'Burglary': True, so we remove 'Burglary': False)
    cpts = [df for df in net.probabilities.values()]
    updated_cpts = []
    for df in cpts:
        for key, value in evidence.items():
            if key in df.columns:
                df = df[df[key]==value]
                updated_cpts.append(df)
            else:
                updated_cpts.append(df)

    #Retrieve product formula:
    product_formula = [list(df.columns[df.columns != 'prob']) for df in updated_cpts]

    # Determine your elimination ordering before you call the run function. The elimination ordering   
    # is either specified by a list or a heuristic function that determines the elimination ordering
    # given the network. Experimentation with different heuristics will earn bonus points. The elimination
    # ordering can for example be set as follows:
    def minimum_factor_size(factor, evidence=set()):
        elim_order = []
        factor_copy = factor.copy()
        variables = {var for factor in factor_copy for var in factor} - set(evidence.keys()) #All variables - evidence variable(s)

        while variables:
            sizes = {}
            for var in variables:
                involved = [set(factor) for factor in factor_copy if var in factor]
                union_involved = set().union(*involved) #Concatenate all variables in involved into one set, removing duplicates and returning 1 single set.
                sizes[var] = 2**len(union_involved) #2^k for boolean operations. 

            best_var = min(sizes, key=sizes.get)
            elim_order.append(best_var)
            variables.remove(best_var)

        return elim_order

    print(minimum_factor_size(product_formula, evidence))
    # Call the variable elimination function for the queried node given the evidence and the elimination ordering as follows:   
    result = ve.run(query, evidence, minimum_factor_size(product_formula, evidence))
    print("-"*200)
    print(f"CPT of variable {query}"); 
    if evidence: print(f'given evidence {evidence}:')
    print(result)