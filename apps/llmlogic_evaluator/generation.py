# Logic for generating propositional logic problems
# Adapted from https://github.com/tammet/llmlog/blob/main/makeproblems.py by Tanel Tammet
# Original License: Apache-2.0

import random
import math
import json  # Added for later output step
import itertools # For iterating through parameter combinations


# ================== Generation Core ==================

def make_prop_problem(varnr, maxlen, ratio, hornflag):
    if varnr < 2:
        return []

    # see the discussion of the ratio 4 in the classic
    # http://www.aaai.org/Papers/AAAI/1992/AAAI92-071.pdf :
    # 4 is the integer closest to the \"hardest\" ratio

    nr = int(varnr * ratio)  # how many 3-element clauses

    # loop until the clause set contains at least one
    # fully negative and one fully positive clause
    while True:
        res = []
        count = 0
        units = {}
        fullneg = False
        fullpos = False
        # loop until enough clauses have been generated
        while count < nr:
            clause = []
            poscount = 0
            negcount = 0
            for j in range(maxlen):
                r1 = random.random()
                if r1 < 0.5:
                    s = 0 - 1
                else:
                    if hornflag and poscount > 0:
                        continue
                    s = 1
                    poscount += 1
                r2 = random.random()
                v = math.floor((r2 * varnr) + 1)
                if s * v not in clause:
                    clause.append(s * v)
                    if s < 0:
                        negcount += 1
            if is_tautology(clause):
                continue
            if len(clause) == 1:
                var = clause[0]
                if 0 - var in units:
                    continue
                if var in units:
                    continue
                units[var] = True
            clause.sort()
            if clause in res:
                continue
            if negcount == len(clause):
                fullneg = True
            if negcount == 0:
                fullpos = True
            res.append(clause)
            count += 1
        if fullneg and fullpos:
            break
    return res


def is_tautology(varlist):
    for el in varlist:
        if el > 0 and 0 - el in varlist:
            return True
    return False


def is_horn(varlist):
    count = 0
    for el in varlist:
        if el > 0:
            if count > 0:
                return False
            else:
                count = 1
    return True


def normalize_problem(problem):
    l = []
    for cl in problem:
        l.append(frozenset(cl))
    pset = set(l)
    lst = list(pset)
    l = []
    for el in lst:
        # Convert frozenset back to list and sort
        l.append(sorted(list(el)))
    s = sorted(l, key=lambda x: (len(x), x))
    return s


# ================== Balancing and Solving (Truth Table) ==================

# wanted: number of problems we want to get
# varnr: number of vars
# maxlen: max length of a clause
# ratio: multiplied with n gives the number of clauses
# hornflag: if true, only horn are OK
def make_balanced_prop_problem_list(wanted, varnr, maxlen, ratio, hornflag):
    true_problems = []
    false_problems = []
    true_models = [] # Store models for true problems
    truecount = 0
    falsecount = 0
    loop_iterations = 0
    while True:
        loop_iterations += 1
        raw_problem = make_prop_problem(varnr, maxlen, ratio, hornflag)
        problem = normalize_problem(raw_problem)
        if not problem:
            continue
        table_res = truth_table_solve(problem)

        if not table_res:
            print(f"Warning: truth_table_solve failed for params: vars={varnr}, len={maxlen}, ratio={ratio}, horn={hornflag}. Skipping iteration.")
            continue

        satisfiable = bool(table_res[0])

        if satisfiable:
            truecount += 1
            if len(true_problems) <= len(false_problems):
                true_problems.append(problem)
                true_models.append(table_res[0]) # Store the model found
        else:
            falsecount += 1
            if len(false_problems) <= len(true_problems):
                false_problems.append(problem)

        if len(true_problems) + len(false_problems) >= wanted:
            break

    # Return counts, problems, AND the models for the true problems
    return ([truecount, falsecount, true_problems, false_problems, true_models])


# --- Truth Table Solver Globals ---
# (Consider refactoring these into a class or passing as arguments if needed elsewhere)
truth_value_leaves_count = 0
truth_value_calc_count = 0
trace_flag = False  # false if no trace
trace_method = "text"  # ok values: text, html or console
trace_list = []  # list of strings
origvars = []  # original variable names to use in trace, if available and not false
result_model = []  # set by solver to a resulting model: values of vars
truth_check_place = "nodes" # Algorithm choice: 'nodes' or 'leaves'

# --- Truth Table Solver Functions ---
def truth_table_solve(clauses):
    maxvar = 0
    for cl in clauses:
        for el in cl:
            if abs(el) > maxvar:
                maxvar = abs(el)
    if maxvar > 20:
        print("Warning: too many variables for truth table solver (max 20):", maxvar)
        return None
    res = search(clauses, maxvar, "nodes", False, {})
    return res


def search(clauses, maxvarnr, algorithm, trace, varnames):
    global truth_check_place, trace_flag, trace_method, origvars
    global truth_value_calc_count, truth_value_leaves_count, trace_list, result_model

    truth_check_place = algorithm
    trace_flag = bool(trace)
    trace_method = trace if trace else None
    origvars = varnames if varnames else False

    truth_value_calc_count = 0
    truth_value_leaves_count = 0
    trace_list = []
    result_model = []

    if maxvarnr is None:
        maxvarnr = 0
        for c in clauses:
            for j in c:
                nr = abs(j)
                if nr > maxvarnr:
                    maxvarnr = nr

    varvals = [0] * (maxvarnr + 1)

    res = (satisfiable_by_table_at(clauses, varvals, 1, 1, 1) or
           satisfiable_by_table_at(clauses, varvals, 1, -1, 1))

    txt = f"finished: evaluations count is {truth_value_calc_count}, leaves count is {truth_value_leaves_count}"
    trace_list.append(txt)

    return [result_model, "\r\n".join(trace_list)] if res else [False, "\r\n".join(trace_list)]


def satisfiable_by_table_at(clauses, varvals, varnr, val, depth):
    global truth_value_leaves_count, trace_flag, truth_check_place

    varvals[varnr] = val
    if varnr == len(varvals) - 1:
        truth_value_leaves_count += 1

    if trace_flag:
        pass

    if truth_check_place != "leaves" or varnr == len(varvals) - 1:
        tmp = clauses_truth_value_at(clauses, varvals, depth)
        if tmp == 1:
            store_model(varvals)
            varvals[varnr] = 0
            return True
        if tmp == -1:
            varvals[varnr] = 0
            return False

    if varnr < len(varvals) - 1:
        if (satisfiable_by_table_at(clauses, varvals, varnr + 1, 1, depth + 1) or
            satisfiable_by_table_at(clauses, varvals, varnr + 1, -1, depth + 1)):
            varvals[varnr] = 0
            return True

    varvals[varnr] = 0
    return False


def clauses_truth_value_at(clauses, varvals, depth):
    global truth_value_calc_count, trace_flag
    truth_value_calc_count += 1
    allclausesfound = True

    for clause in clauses:
        clauseval = 0
        allvarsfound = True

        for nr in clause:
            polarity = 1
            if nr < 0:
                nr = -nr
                polarity = -1
            vval = varvals[nr]
            if vval == polarity:
                clauseval = 1
                break
            elif vval == 0:
                allvarsfound = False

        if clauseval != 1 and allvarsfound:
            if trace_flag:
                pass
            return -1

        if not allvarsfound:
            allclausesfound = False

    if allclausesfound:
        if trace_flag:
            pass
        return 1

    if trace_flag:
        pass
    return 0

def store_model(varvals):
    global result_model
    result_model.clear()
    for i in range(1, len(varvals)):
        if varvals[i] != 0:
            result_model.append(i * varvals[i])


# ================== Resolution Prover & Proof Generation ==================

# --- Resolution Solver Globals ---
# (Consider refactoring into a class)
usablecls_maxlen = 100 # Max clause length considered by basic solver
lastclid = 0
usablecls = []
usablecls_bylen = [] # List of lists, indexed by length
allcls = {} # Dictionary mapping clid -> clause structure

# --- Resolution Solver Functions ---

# solve_prop_problem returns a proof clause (empty list if contradiction) if proof is found,
# else None if satisfiable
def solve_prop_problem(clauses, resolution_strategy: str = 'original'):
    global lastclid, usablecls, usablecls_bylen, allcls
    # Initialize globals for this run
    selected_clauses_count = 0
    processed_clauses = []
    lastclid = 0
    usablecls = []
    allcls = {}
    usablecls_bylen = []
    for i in range(usablecls_maxlen + 1):
        usablecls_bylen.append([])

    # Prepare initial clauses
    initial_clause_ids = []
    for cl_idx, cl in enumerate(clauses):
        varset = frozenset(cl) # Use frozenset for hashability/set ops
        if is_tautology_set(varset):
            continue
        lastclid += 1
        newcl = [lastclid, None, varset]
        add_usable(newcl)
        initial_clause_ids.append(lastclid)

    # Main resolution loop
    result = None  # Becomes the contradiction clause if found
    loop_count = 0
    while True:
        loop_count += 1
        selected = select_usable()
        if not selected:
            return None # Satisfiable

        if is_tautology_set(selected[2]):
            continue

        # --- Subsumption Check ---
        found_subsumer = False
        for oldclause in processed_clauses:
            if naive_subsumed(oldclause, selected):
                found_subsumer = True
                break
        if found_subsumer:
            continue
        # --- End Subsumption Check ---

        selected_clauses_count += 1
        # Resolve selected against all previously processed clauses
        for processed in processed_clauses:
            # --- Choose resolution strategy ---
            if resolution_strategy == 'standard':
                contra_cl = _do_resolution_steps_standard(selected, processed)
            else: # Default to 'original'
                contra_cl = _do_resolution_steps_original_heuristic(selected, processed)
            # --- End strategy choice ---

            if contra_cl:
                result = contra_cl
                break # Exit inner loop (processed clauses)
        if result:
            break # Exit outer loop (while True)
        processed_clauses.append(selected)

    return result # Returns the final contradiction clause [clid, hist, frozenset()]

def make_internal_cl(cl):
    global lastclid
    lastclid += 1
    newcl = [lastclid, None, frozenset(cl)] # Use frozenset
    return newcl

# Renamed function containing original makeproblems.py logic
def _do_resolution_steps_original_heuristic(clx, cly):
    # Matches original makeproblems.py logic (lines 442-470)
    # Clause format: [id, history, frozenset_of_literals]
    global lastclid
    clxels = clx[2]
    clyels = cly[2]

    if not clxels or not clyels:
        return None

    try:
        clxmin = min(clxels)
        clymin = min(clyels)
    except ValueError:
        return None

    if clxmin < 0 and clymin > 0:
        cutvar = clxmin
        cl = clyels
    elif clymin < 0 and clxmin > 0:
        cutvar = clymin
        cl = clxels
    else:
        return None

    if 0 - cutvar in cl:
        newraw = (clxels.union(clyels)) - {cutvar, 0 - cutvar}
        if is_tautology_set(newraw):
            return None

        lastclid += 1
        newcl = [lastclid, [clx[0], cly[0]], newraw]
        add_usable(newcl)

        if not newraw:
            return newcl
        else:
            return None
    else:
        return None

# Added back: Standard resolution step implementation
def _do_resolution_steps_standard(clx, cly):
    # Standard approach: iterates through all literals
    # Clause format: [id, history, frozenset_of_literals]
    global lastclid
    clxels = clx[2]
    clyels = cly[2]
    found_contradiction = None

    for lit in clxels:
        if -lit in clyels:
            newraw = (clxels.union(clyels)) - {lit, -lit}
            if is_tautology_set(newraw):
                continue

            lastclid += 1
            newcl = [lastclid, [clx[0], cly[0]], newraw]
            add_usable(newcl)
            if not newraw:
                found_contradiction = newcl
                break # Stop checking other literals for this pair

    return found_contradiction

def add_usable(cl):
    global usablecls, usablecls_bylen, allcls
    if cl[0] in allcls:
        return

    usablecls.append(cl)
    allcls[cl[0]] = cl
    l = len(cl[2])
    if l >= usablecls_maxlen:
        usablecls_bylen[usablecls_maxlen].append(cl)
    else:
        usablecls_bylen[l].append(cl)


def select_usable():
    global usablecls_bylen
    for i in range(len(usablecls_bylen)):
        if usablecls_bylen[i]:
            selected = usablecls_bylen[i].pop()
            return selected
    return None # No usable clauses left

def is_tautology_set(varset):
    for el in varset:
        if el > 0 and 0 - el in varset:
            return True
    return False


def naive_subsumed(general, special):
    if general[2].issubset(special[2]):
        return True
    else:
        return False

def clause_to_str(cl):
    s = ""
    lst = sorted(list(cl[2]))
    for i in range(len(lst)):
        s += str(lst[i]) + " "
    s = str(cl[0]) + ": " + s
    return s

def print_trace(depth, x):
    pass

# --- Proof Reconstruction Functions ---
def makeproof(resolve_res, allcls_from_run):
    if not resolve_res or type(resolve_res) != list or resolve_res[2]:
        return []

    proofcls = {}
    makeproof_aux(resolve_res, allcls_from_run, proofcls)

    lst = sorted(proofcls.values(), key=lambda x: x[0])

    nr = 0
    nrs = {}
    for el in lst:
        if el[0] not in nrs:
            nr += 1
            nrs[el[0]] = nr

    lst2 = []
    for el in lst:
        newhist = []
        if el[1]:
            for hel in el[1]:
                if hel in nrs:
                    newhist.append(nrs[hel])
        newcl = [nrs[el[0]], newhist, sorted(list(el[2]))]
        lst2.append(newcl)

    return lst2

def makeproof_aux(incl, allcls_from_run, proofcls):
    if not incl or type(incl) != list:
        return
    if incl[0] in proofcls:
        return

    proofcls[incl[0]] = incl

    if incl[1]:
        for parent_id in incl[1]:
            if parent_id in allcls_from_run:
                parent_cl = allcls_from_run[parent_id]
                makeproof_aux(parent_cl, allcls_from_run, proofcls)


# --- Horn Clause Solver Functions ---

# solve_prop_horn_problem returns a list of derived units (positive literals).
# If the list contains 0, it means contradiction was derived.
def solve_prop_horn_problem(inclauses):
    newunits = []  # List of new units derived during one iteration
    posunits = {}  # Set of derived positive units (lit -> True)
    hornrules = []  # List of horn clauses (potential rules)
    derivedunits = [] # Result list of derived units in order

    # Process input clauses
    clauses = [sorted(cl) for cl in inclauses] # Ensure internal sorting
    clauses.sort(key=lambda x: (len(x), x)) # Sort clauses themselves

    # Separate initial units and potential horn rules
    for cl in clauses:
        pos_lits = [lit for lit in cl if lit > 0]
        is_potential_horn = len(pos_lits) <= 1

        if len(cl) == 1 and cl[0] > 0:
            # Initial positive unit clause
            v = cl[0]
            if v not in posunits:
                posunits[v] = True
                newunits.append(v)
                derivedunits.append(v) # Add initial units to derived list
        elif is_potential_horn:
            # Store clauses with 0 or 1 positive literals
            hornrules.append(cl)
        # else: Non-Horn clause, ignore in this solver

    # Iterative unit propagation
    while newunits:
        nextunits = [] # Units derived in this iteration
        processed_rules_in_iter = set() # Avoid reusing same rule multiple times with same trigger unit

        for unit in newunits:
            negunit = -unit
            rules_to_remove = []
            temp_hornrules = list(hornrules) # Iterate over a copy if modifying

            for i, rule in enumerate(temp_hornrules):
                if i in processed_rules_in_iter:
                    continue

                if negunit in rule:
                    # Check if this unit satisfies a negative literal in the rule
                    all_neg_lits_satisfied = True
                    derived_pos_lit = None
                    has_pos_lit = False

                    for lit in rule:
                        if lit < 0:
                            if lit != negunit and -lit not in posunits:
                                all_neg_lits_satisfied = False
                                break
                        else: # lit > 0 (positive literal)
                            has_pos_lit = True
                            derived_pos_lit = lit
                            # If this positive lit is already known, rule is satisfied, not useful for derivation
                            if lit in posunits:
                                all_neg_lits_satisfied = False 
                                break 

                    if all_neg_lits_satisfied:
                        processed_rules_in_iter.add(i)
                        # If we reached here, all negative literals are satisfied by known units
                        if has_pos_lit:
                            # Derive the positive literal
                            if derived_pos_lit not in posunits:
                                posunits[derived_pos_lit] = True
                                nextunits.append(derived_pos_lit)
                                derivedunits.append(derived_pos_lit)
                        else:
                            # Contradiction: all negative literals satisfied, no positive literal means empty clause derived
                            derivedunits.append(0)
                            return derivedunits # Contradiction found

            # Optional: remove satisfied rules (be careful with indices if modifying hornrules directly)
            # for idx in sorted(rules_to_remove, reverse=True):
            #     hornrules.pop(idx)

        newunits = nextunits # Prepare for next iteration

    return derivedunits # Return list of derived units (without 0 if no contradiction)

# ================== Generation Configuration ==================

# Default ranges and settings, can be overridden by kwargs in run_generation
DEFAULT_VARNR_RANGE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
DEFAULT_CL_LEN_RANGE = [3, 4]
DEFAULT_HORN_FLAGS = [True, False]
DEFAULT_PROBS_PER_CASE = 20 # Must be even

# Ratios for problem generation complexity based on clause length and variable count
# Extracted from makeproblems.py
GOOD_RATIOS = {
    # max_clause_len: [general_ratio_list_by_varnr, horn_ratio_scalar]
    # General ratio list indices correspond to varnr (index 0,1,2 unused)
    2: [[], 1.9, 1.3], # Original script didn't seem to use len=2, added placeholder
    3: [[0,0,0, 2.7, 3.4, 3.9, 3.5, 4.2, 4.0, 4.0, 4.0, 4.0, 4.3, 4.0], 2.0],
    4: [[0,0,0, 3.2, 4.4, 5.6, 6.4, 6.9, 6.7, 7.6, 7.6, 7.8, 7.6], 3.1],
    5: [[0,0,0, 3.3, 5.5, 7.7, 9.4, 10.8, 11.6, 12.4, 12.9, 13.9, 14.1], 4.6]
    # Note: Ratios might need adjustment or recalculation depending on exact needs.
    # The lists in the original were derived experimentally.
}

# ================== Main Generation Function ==================

def run_generation(output_file: str, count: int, resolution_strategy: str = 'original', **kwargs):
    """Generates a dataset of logic problems and saves it to a JSON Lines file.

    Iterates through combinations of variable numbers, clause lengths, and horn flags,
    generating a balanced set of satisfiable and unsatisfiable problems for each.

    Args:
        output_file: Path to the output JSON Lines file.
        count: Total number of problems *per configuration case* (must be even).
        resolution_strategy: 'original' (fast heuristic) or 'standard' (thorough).
        **kwargs: Optional generation parameters overriding defaults:
            var_range (list[int]): Range of variable numbers.
            len_range (list[int]): Range of max clause lengths.
            horn_flags (list[bool]): List of horn flags (True, False, or both).
    """
    global allcls # Needed for makeproof

    if count % 2 != 0:
        raise ValueError("Count must be an even number for balanced generation.")
    probs_for_onecase = count

    # Get parameters from kwargs or use defaults
    varnr_range = kwargs.get('var_range', DEFAULT_VARNR_RANGE)
    cl_len_range = kwargs.get('len_range', DEFAULT_CL_LEN_RANGE)
    horn_flags = kwargs.get('horn_flags', DEFAULT_HORN_FLAGS)

    print(
        f"Starting problem generation to {output_file}...\n"+
        f"  Var Counts: {varnr_range}\n"+
        f"  Clause Lengths: {cl_len_range}\n"+
        f"  Horn Flags: {horn_flags}\n"+
        f"  Problems per Case: {probs_for_onecase}\n"+
        f"  Resolution Strategy: {resolution_strategy}" # Added strategy info
    )

    probnr = 0
    all_problems_data = []

    # Iterate through all combinations of parameters
    for varnr, cllen, hornflag in itertools.product(varnr_range, cl_len_range, horn_flags):
        print(f"Generating case: vars={varnr}, len={cllen}, horn={hornflag}")

        # Determine the generation ratio
        if cllen not in GOOD_RATIOS:
            print(f"Warning: No predefined ratio for clause length {cllen}. Skipping case.")
            continue
        ratios_for_len = GOOD_RATIOS[cllen]
        if hornflag:
            ratio = ratios_for_len[1] # Scalar horn ratio
        else:
            general_ratios = ratios_for_len[0] # List of general ratios by varnr
            if varnr >= len(general_ratios):
                ratio = general_ratios[-1] # Use last available ratio if varnr exceeds list
            else:
                ratio = general_ratios[varnr]
            if ratio == 0: # Handle cases where ratio might be 0 in the config
                 print(f"Warning: Ratio is 0 for vars={varnr}, len={cllen}, general. Skipping case.")
                 continue

        # Generate a balanced list for this case
        # Returns [truecount, falsecount, true_problems_list, false_problems_list, true_models_list]
        problst_result = make_balanced_prop_problem_list(probs_for_onecase, varnr, cllen, ratio, hornflag)

        if problst_result is None or len(problst_result) < 5:
            print(f"Error generating balanced list for case: vars={varnr}, len={cllen}, horn={hornflag}. Skipping.")
            continue

        truelist = problst_result[2]
        falselist = problst_result[3]
        true_models = problst_result[4] # Get the models for true problems

        # Interleave true and false problems for output
        choosefrom_true = True
        processed_count = 0
        true_model_index = 0 # Index for retrieving pre-computed models
        while truelist or falselist:
            processed_count += 1
            problem_clauses = None
            is_satisfiable = None
            proof_or_model = None
            horn_derived_units = None

            if choosefrom_true:
                if truelist:
                    problem_clauses = truelist.pop(0)
                    # Retrieve the corresponding pre-computed model
                    if true_model_index < len(true_models):
                        proof_or_model = true_models[true_model_index]
                        true_model_index += 1
                    else:
                        # Should not happen if lists are balanced correctly
                        print(f"Warning: Mismatch between truelist and true_models for case {varnr},{cllen},{hornflag}")
                        proof_or_model = [] # Fallback
                    is_satisfiable = True
                else:
                    choosefrom_true = not choosefrom_true
                    continue
            else:
                if falselist:
                    problem_clauses = falselist.pop(0)
                    is_satisfiable = False
                    # Run resolution prover with the chosen strategy
                    resolve_res = solve_prop_problem(problem_clauses, resolution_strategy=resolution_strategy)
                    if resolve_res:
                        current_allcls_copy = allcls.copy()
                        proof_or_model = makeproof(resolve_res, current_allcls_copy)
                    else:
                        print(f"Warning: Resolution failed to prove unsatisfiability for supposedly false problem {probnr+1} with strategy '{resolution_strategy}'")
                        proof_or_model = []
                else:
                    choosefrom_true = not choosefrom_true
                    continue

            if problem_clauses is not None:
                probnr += 1
                horn_derived_units = solve_prop_horn_problem(problem_clauses)

                # Structure output data as a dictionary
                problem_data = {
                    "id": probnr,
                    "max_vars": varnr,
                    "max_clause_len": cllen,
                    "is_horn_intended": hornflag,
                    "is_satisfiable": is_satisfiable,
                    "problem_clauses": problem_clauses,
                    "proof_or_model": proof_or_model,
                    "horn_derived_units": horn_derived_units
                }
                all_problems_data.append(problem_data)

            # Alternate choice for next iteration
            choosefrom_true = not choosefrom_true

    # Write all collected problems to the output file
    try:
        with open(output_file, 'w') as f:
            for problem_entry in all_problems_data:
                json.dump(problem_entry, f)
                f.write('\n')
        print(f"Successfully generated {probnr} problems to {output_file}")
    except IOError as e:
        print(f"Error writing problems to file {output_file}: {e}")

# Remove placeholder comment inside the function body if it exists
# # TODO: Implement logic adapted from makeproblems.py ... 