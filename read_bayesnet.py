"""
Enhanced Bayesian network parser.

Handles both the standard BIF format (earthquake.bif style with | separator)
and the table-based BIF format (endorisk.bif style with quoted variable names).

Based on original code by Joris van Vugt, Moira Berens, Leonieke van den Bulk.
"""

import pandas as pd
import re
from itertools import product as iter_product


class BayesNet:
    """
    Represents a Bayesian network parsed from a .bif file.
    Supports both standard BIF format and the pgmpy table-based format.
    Uses pandas DataFrames for conditional probability tables.
    """

    def __init__(self, filename=None):
        self.name = ""
        self.values = {}
        self.probabilities = {}
        self.parents = {}

        if filename:
            self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as f:
            content = f.read()

        if self._is_table_format(content):
            self._parse_table_format(content)
        else:
            self._parse_standard_format(filename)

    def _is_table_format(self, content):
        return bool(re.search(r'variable\s+"', content))

    # ── Table-based format (endorisk.bif / endomcancer.bif) ──────────────

    def _parse_table_format(self, content):
        lines = content.split('\n')

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('network'):
                match = re.search(r'network\s+"?([^"{}]+)"?\s*\{', stripped)
                if match:
                    self.name = match.group(1).strip()
                break

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith('variable'):
                match = re.search(r'variable\s+"([^"]+)"', stripped)
                if match:
                    var_name = match.group(1)
                    i += 1
                    while i < len(lines):
                        type_line = lines[i].strip()
                        if type_line.startswith('type'):
                            vals = re.findall(r'"([^"]+)"', type_line)
                            self.values[var_name] = vals
                            self.parents[var_name] = []
                            break
                        i += 1
            i += 1

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith('probability'):
                match = re.search(r'probability\s*\(([^)]+)\)', stripped)
                if match:
                    var_names = re.findall(r'"([^"]+)"', match.group(1))
                    child = var_names[0]
                    parents = var_names[1:] if len(var_names) > 1 else []
                    self.parents[child] = parents

                    i += 1
                    while i < len(lines):
                        table_line = lines[i].strip()
                        if table_line.startswith('table'):
                            prob_str = table_line.replace('table', '').replace(';', '').strip()
                            probs = [float(p) for p in prob_str.split()]
                            self._build_cpt_from_table(child, parents, probs)
                            break
                        elif '}' in table_line:
                            break
                        i += 1
            i += 1

    def _build_cpt_from_table(self, child, parents, probs):
        """Build CPT DataFrame from flat table values.
        In pgmpy BIF format the child cycles slowest, last parent cycles fastest.
        """
        all_vars = [child] + parents
        all_values = [self.values[v] for v in all_vars]
        combos = list(iter_product(*all_values))

        columns = [child] + parents + ['prob']
        rows = [list(combo) + [p] for combo, p in zip(combos, probs)]
        self.probabilities[child] = pd.DataFrame(rows, columns=columns)

    # ── Standard format (earthquake.bif) ─────────────────────────────────

    def _parse_standard_format(self, filename):
        with open(filename, 'r') as f:
            all_lines = f.readlines()

        for line_number, line in enumerate(all_lines):
            if line.startswith('network'):
                self.name = ' '.join(line.split()[1:-1])
            elif line.startswith('variable'):
                self._parse_variable_std(line_number, all_lines)
            elif line.startswith('probability'):
                self._parse_probability_std(line_number, all_lines)

    def _parse_variable_std(self, line_number, all_lines):
        variable = all_lines[line_number].split()[1]
        line = all_lines[line_number + 1]
        start = line.find('{') + 1
        end = line.find('}')
        values = [v.strip() for v in line[start:end].split(',')]
        self.values[variable] = values
        self.parents[variable] = []

    def _parse_probability_std(self, line_number, all_lines):
        line = all_lines[line_number]
        variable, parents = self._parse_parents_std(line)
        next_line = all_lines[line_number + 1].strip()

        if next_line.startswith('table'):
            comma_sep_probs = next_line.split('table')[1].split(';')[0].strip()
            probs = [float(p) for p in comma_sep_probs.split(',')]
            df = pd.DataFrame(columns=[variable, 'prob'])
            for value, p in zip(self.values[variable], probs):
                df.loc[len(df)] = [value, p]
            self.probabilities[variable] = df
        else:
            df = pd.DataFrame(columns=[variable] + parents + ['prob'])
            for l in all_lines[line_number + 1:]:
                if '}' in l:
                    break
                comma_sep_values = l.split('(')[1].split(')')[0]
                values = [v.strip() for v in comma_sep_values.split(',')]
                comma_sep_probs = l.split(')')[1].split(';')[0].strip()
                probs = [float(p) for p in comma_sep_probs.split(',')]
                for value, p in zip(self.values[variable], probs):
                    df.loc[len(df)] = [value] + values + [p]
            self.probabilities[variable] = df

    def _parse_parents_std(self, line):
        start = line.find('(') + 1
        end = line.find(')')
        variables = line[start:end].strip().split('|')
        variable = variables[0].strip()
        if len(variables) > 1:
            self.parents[variable] = [v.strip() for v in variables[1].split(',')]
        else:
            self.parents[variable] = []
        return variable, self.parents[variable]

    # ── Utility methods ──────────────────────────────────────────────────

    @property
    def nodes(self):
        return list(self.values.keys())

    def get_children(self, variable):
        return [v for v in self.nodes if variable in self.parents.get(v, [])]

    def save_bif(self, filename):
        """Save the network in table-based BIF format."""
        with open(filename, 'w') as f:
            f.write(f'// Bayesian network\n')
            f.write(f'network "{self.name}" {{\n}}\n')

            for var in self.nodes:
                vals = self.values[var]
                f.write(f'variable  "{var}" {{ // {len(vals)} values\n')
                vals_str = '  '.join(f'"{v}"' for v in vals)
                f.write(f'\ttype discrete[{len(vals)}] {{  {vals_str} }};\n')
                f.write(f'\tproperty "position = (40, 475)" ;\n')
                f.write(f'}}\n')

            for var in self.nodes:
                parents = self.parents[var]
                all_vars = [var] + parents
                vars_str = '  '.join(f'"{v}"' for v in all_vars)

                all_values = [self.values[v] for v in all_vars]
                combos = list(iter_product(*all_values))
                n_total = len(combos)

                df = self.probabilities[var]
                probs = []
                for combo in combos:
                    mask = pd.Series([True] * len(df))
                    for v, val in zip(all_vars, combo):
                        mask &= (df[v] == val)
                    prob_val = df.loc[mask, 'prob'].values
                    probs.append(float(prob_val[0]) if len(prob_val) > 0 else 0.0)

                f.write(f'probability (  {vars_str} ) {{ // {len(all_vars)} variable(s) and {n_total} values\n')
                probs_str = ' '.join(str(p) for p in probs)
                f.write(f'\ttable {probs_str} ;\n')
                f.write(f'}}\n')
