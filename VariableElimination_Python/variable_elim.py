"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Class for the implementation of the variable elimination algorithm.

"""

class VariableElimination():

    def __init__(self, network):
        """
        Initialize the variable elimination algorithm with the specified network.
        Add more initializations if necessary.

        """
        self.network = network

    def run(self, query, observed_cpts, elim_order, factor_product_function):
        """
        Use the variable elimination algorithm to find out the probability
        distribution of the query variable given the observed variables

        Input:
            query:      The query variable
            observed:   A dictionary of the observed variables {variable: value}
            elim_order: Either a list specifying the elimination ordering
                        or a function that will determine an elimination ordering
                        given the network during the run

        Output: A variable holding the probability distribution
                for the query variable

        """

        #Compute variable elimination:
        for element in elim_order:
            if element == query:
                continue
            factors_with_element = [df for df in observed_cpts if element in df.columns]
            factors_without_element = [df for df in observed_cpts if element not in df.columns]

            if not factors_with_element:
                continue 

            combined = factors_with_element[0]

            for df in factors_with_element[1::]:
                common = [column for column in combined.columns if column in df.columns and column !='prob']
                combined = (combined.merge(df, on=common).assign(prob=lambda x: x.prob_x*x.prob_y).drop(columns=["prob_x", "prob_y"]))

            keep = [column for column in combined.columns if column not in {element, 'prob'}] #Keep columns that do not consist of target column (following from elim_order)
            new_factor = combined.groupby(keep, as_index=False)['prob'].sum()
            
            observed_cpts = factors_without_element + [new_factor]
            factor_product_function = [[var for var in vars_ if var != element] for vars_ in factor_product_function]

        #Sum out remaining vectors containing query variable
        cpt_query = combined.groupby(query, as_index=False)['prob'].sum()
        
        #Marginalize
        sum_rows = cpt_query['prob'].sum()
        marginalized_cpt_query = cpt_query.copy()
        marginalized_cpt_query['prob'] /= sum_rows
        return marginalized_cpt_query