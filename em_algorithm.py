import pandas as pd
import random
import math
from itertools import product as iter_product

#Redefine the node names to be more readable 
columns_renamed = {
    'Progesteron receptor': 'PR',
    'Survival 1yr': 'Survival1yr',
    'Lymph node metastasis': 'LNM',
    'Postoperative grade': 'Histology',
    'Preoperative grade': 'PrimaryTumor',
    'Adjuvant therapy': 'Therapy',
    'Survival 3yr': 'Survival3yr',
    'Myometrial invasion': 'MyometrialInvasion',
    'Estrogen receptor': 'ER',
    'Endometrium in cervical cytology': 'Cytology',
    'Survival 5yr': 'Survival5yr',
    'Enlarged nodes CT': 'CTMRI',
}

def load_data(filepath, network):
    data = pd.read_csv(filepath, sep='\t', keep_default_na=False)
    data = data.rename(columns=columns_renamed)

    #Check all network nodes are in the data columns
    for node in network.nodes:
        if node not in data.columns:
            raise ValueError(f"Network node '{node}' not found in data columns")

    #Keep only columns that are network nodes
    data = data[[col for col in data.columns if col in network.nodes]]
    return data

def randomize_cpts(network):
    """Randomize the CPTs of the network"""
    for var in network.nodes:
        parents = network.parents[var]
        values = network.values[var]
        cpt = network.probabilities[var]

        if parents:
            parent_combos = cpt[parents].drop_duplicates().values.tolist()
            rows = []
            for parent_vals in parent_combos:
                weights = [random.random() for _ in values]
                total = sum(weights)
                probs = [w / total for w in weights]
                for val, prob in zip(values, probs):
                    rows.append([val] + parent_vals + [prob])
            network.probabilities[var] = pd.DataFrame(rows, columns=[var] + parents + ['prob'])
        else:
            weights = [random.random() for _ in values]
            total = sum(weights)
            probs = [w / total for w in weights]
            rows = [[val, prob] for val, prob in zip(values, probs)]
            network.probabilities[var] = pd.DataFrame(rows, columns=[var, 'prob'])


class EMAlgorithm:
    def __init__(self, network, data, log=None):
        self.network = network
        self.data = data
        self.log = log

    def write(self, msg):
        print(msg)
        if self.log:
            print(msg, file=self.log)

    #Helper functions:
    def _lookup_prob(self, var, assignment):
        """Look up P(var=val | parents=parent_vals) from the current CPTs."""
        cpt = self.network.probabilities[var]
        parents = self.network.parents[var]
        mask = (cpt[var] == assignment[var])
        for p in parents:
            mask &= (cpt[p] == assignment[p])
        result = cpt.loc[mask, 'prob']
        if len(result) == 0:
            return 0.0
        return float(result.iloc[0])

    def _joint_probability(self, assignment):
        """Compute P(x1,...,xn), so the product of all CPT lookups."""
        prob = 1.0
        for var in self.network.nodes:
            prob *= self._lookup_prob(var, assignment)
            if prob == 0:
                break
        return prob
        
    def _add_counts(self, expected_counts, assignment, weight):
        for var in self.network.nodes:
            parents = self.network.parents[var]
            all_vars = [var] + parents
            counts = expected_counts[var]
            mask = pd.Series([True] * len(counts))
            for v in all_vars:
                mask &= (counts[v] == assignment[v])
            expected_counts[var].loc[mask, 'count'] += weight

    #EM Algorithm:
    def e_step(self):
        #Initialize empty counts for each variable's CPT
        expected_counts = {}

        for var in self.network.nodes:
            parents = self.network.parents[var]
            all_vars = [var] + parents
            all_values = [self.network.values[v] for v in all_vars]
            combos = list(iter_product(*all_values))
            rows = [list(combo) + [0.0] for combo in combos]
            expected_counts[var] = pd.DataFrame(rows, columns=all_vars + ['count'])

        loglikelihood = 0.0 
        #Find which variables are hidden (empty in every row)
        hidden_vars = [col for col in self.network.nodes
                    if (self.data[col] == '').all()]
        hidden_combos = list(iter_product(*[self.network.values[h] for h in hidden_vars]))

        #Loop over every row
        for idx, row in self.data.iterrows():
            observed = {col: row[col] for col in self.data.columns
                        if col in self.network.nodes and row[col] != ''}
            if not hidden_vars:
                #Complete row --> count directly:
                prob = self._joint_probability(observed)
                if prob > 0:
                    loglikelihood += math.log(prob)
                self._add_counts(expected_counts, observed, 1.0)
            else:
                #Incomplete row -->  compute P(hidden | observed) for each combo:
                probs = []
                assignments = []
                for combo in hidden_combos:
                    assignment = dict(observed)
                    for var, val in zip(hidden_vars, combo):
                        assignment[var] = val
                    p = self._joint_probability(assignment)
                    probs.append(p)
                    assignments.append(assignment)
                total = sum(probs)
                if total > 0:
                    loglikelihood += math.log(total)
                    probs = [p / total for p in probs]
                for assignment, prob in zip(assignments, probs):
                    self._add_counts(expected_counts, assignment, prob)
        return expected_counts, loglikelihood

    def m_step(self, expected_counts, smoothing=1):
        for var in self.network.nodes:
            parents = self.network.parents[var]
            counts = expected_counts[var]

            if parents:
                totals = counts.groupby(parents, as_index=False)['count'].sum()
                totals = totals.rename(columns={'count': 'total'})
                merged = counts.merge(totals, on=parents)
                n_vals = len(self.network.values[var])
                merged['prob'] = (merged['count'] + smoothing) / (merged['total'] + smoothing * n_vals)
                self.network.probabilities[var] = merged[[var] + parents + ['prob']].copy()
            else:
                total = counts['count'].sum()
                n_vals = len(self.network.values[var])
                counts = counts.copy()
                counts['prob'] = (counts['count'] + smoothing) / (total + smoothing * n_vals)
                self.network.probabilities[var] = counts[[var, 'prob']].copy()

    def run(self, max_iter=50, least_change=1e-4, smoothing=1):
        self.write("=" * 60)
        self.write("EXPECTATION-MAXIMIZATION ALGORITHM")
        self.write(f"Data: {len(self.data)} rows, {len(self.data.columns)} columns")
        hidden = [col for col in self.network.nodes
                    if (self.data[col] == '').all()] #Hidden variables are variables that are not observed in the data, displayed as '' in the code.   
        self.write(f"Hidden variables: {hidden}")
        self.write(f"Max iterations: {max_iter}, Convergence threshold: {least_change}")
        self.write("-" * 60)

        randomize_cpts(self.network)
        self.write("Succesfully initialized CPTs randomly")

        prev_loglikelihood = float('-inf')
        for iteration in range(1, max_iter + 1):
            expected_counts, loglikelihood = self.e_step()
            self.write(f"E-step completed: log-likelihood = {loglikelihood:.4f}")

            self.m_step(expected_counts, smoothing)
            self.write(f"M-step completed: updated CPTs for {len(self.network.nodes)} variables")

            self.write(f"Iteration {iteration}: log-likelihood = {loglikelihood:.4f}")

            if abs(loglikelihood - prev_loglikelihood) < least_change:
                self.write(f"Converged after {iteration} iterations")
                break
            prev_loglikelihood = loglikelihood

        self.write("\nLearned CPTs for hidden variables:")
        for var in hidden:
            self.write(f"\n{var}:")
            self.write(self.network.probabilities[var].to_string(index=False))

        self.write("=" * 60)
        return self.network, loglikelihood