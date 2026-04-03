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

    def _multiply_factors(self, factors):
        """
        Multiplies a list of factors together.
        """
        combined = factors[0]

        for df in factors[1::]:
            common = [column for column in combined.columns if column in df.columns and column !='prob']
            if common:
                combined = combined.merge(df, on=common)
            else:
                combined = combined.merge(df, how='cross')
                
            combined = (combined.assign(prob=lambda x: x.prob_x*x.prob_y).drop(columns=["prob_x", "prob_y"]))

        return combined

    def run(self, query, observed_cpts, elim_order, evidence=None, log=None):
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
        def write(msg):
            print(msg)
            if log:
                print(msg, file=log)
        
        write("VARIABLE ELIMINATION LOG FILE:")
        write(f"\nQuery variable: {query}")
        write(f"Elimination order: {elim_order}")
        write(f"Evidence: {evidence}")
        write("Initial factors:")
        for i, df in enumerate(observed_cpts):
            cols = [col for col in df.columns if col != "prob"]
            write(f"  f{i}: {cols}")
        write("-"*60)


        #Compute variable elimination:
        for element in elim_order:
            if element == query:
                continue
            factors_with_element = [df for df in observed_cpts if element in df.columns]
            factors_without_element = [df for df in observed_cpts if element not in df.columns]

            if not factors_with_element:
                continue 

            combined = self._multiply_factors(factors_with_element)
            write(f"\nEliminating {element}")

            keep = [column for column in combined.columns if column not in {element, 'prob'}] #Keep columns that do not consist of target column (following from elim_order)
            combined = combined.groupby(keep, as_index=False)['prob'].sum()
            write(f"\nAfter summing out {element}: {[col for col in combined.columns if col != 'prob']}") 

            observed_cpts = factors_without_element + [combined]

        result = self._multiply_factors(observed_cpts)

        #Sum out remaining vectors containing query variable
        cpt_query = result.groupby(query, as_index=False)['prob'].sum()
        
        #Marginalize
        sum_rows = cpt_query['prob'].sum()
        cpt_query['prob'] /= sum_rows
        write(f"Final result for query variable {query}:")
        write(f"{cpt_query}")
        write("=" * 60)
        return cpt_query

    #####################################################################################################################

    def run_with_map(self, map_variables, observed_cpts, elim_order, evidence=None, log=None):
        """
        Use the variable elimination algorithm to find out the most probable value of the query variable, given the observed variables and the elimination order.
        """
        def write(msg):
            print(msg)
            if log:
                print(msg, file=log)
        
        write("VARIABLE ELIMINATION WITH MAP LOG FILE:")
        write(f"\nMAP variables: {map_variables}")
        write(f"Elimination order: {elim_order}")
        write(f"Evidence: {evidence}")
        write("Initial factors:")
        for i, df in enumerate(observed_cpts):
            cols = [col for col in df.columns if col != "prob"]
            write(f"  f{i}: {cols}")
        write("-"*60)

        write("\n Step 1: Summing out variables")
        for element in elim_order:
            if element in map_variables:
                continue

            factors_with = [df for df in observed_cpts if element in df.columns]
            factors_without = [df for df in observed_cpts if element not in df.columns]

            if not factors_with:
                continue

            combined = self._multiply_factors(factors_with)

            keep = [c for c in combined.columns if c not in {element, 'prob'}]
            new_factor = combined.groupby(keep, as_index=False)['prob'].sum()
            write(f"  Summed out '{element}' -> factor over {[c for c in new_factor.columns if c != 'prob']}")

            observed_cpts = factors_without + [new_factor]
        
        #Maximize out variables in map_variables
        write("\n Step 2: Maximizing MAP variables")
        traceback = {}

        for var in map_variables:
            factors_with = [df for df in observed_cpts if var in df.columns]
            factors_without = [df for df in observed_cpts if var not in df.columns]

            if not factors_with:
                continue

            combined = self._multiply_factors(factors_with)
            traceback[var] = combined.copy()  # SAVE for traceback

            # Maximize out var (keep row with highest prob per group)
            keep = [c for c in combined.columns if c not in {var, 'prob'}]
            if keep:
                idx = combined.groupby(keep)['prob'].idxmax()
                new_factor = combined.loc[idx, keep + ['prob']].reset_index(drop=True)
            else:
                max_idx = combined['prob'].idxmax()
                new_factor = combined.iloc[[max_idx]][['prob']].reset_index(drop=True)

            write(f"  Maximized '{var}' -> factor over {[c for c in new_factor.columns if c != 'prob']}")
            observed_cpts = factors_without + [new_factor]

        write("\n Step 3: Traceback")
        assignment = {}

        for var in reversed(map_variables):
            if var not in traceback:
                continue
            factor = traceback[var]

            # Fix values of variables we already determined
            for found_var, found_val in assignment.items():
                if found_var in factor.columns:
                    factor = factor[factor[found_var] == found_val]

            best_idx = factor['prob'].idxmax()
            assignment[var] = factor.loc[best_idx, var]
            write(f"  {var} = {assignment[var]}")

        write(f"\nFinal MAP result: {assignment}")
        write("=" * 60)
        return assignment