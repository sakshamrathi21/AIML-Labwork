import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


''' Do not change anything in this function '''
def generate_random_profiles(num_voters, num_candidates):
    '''
        Generates a NumPy array where row i denotes the strict preference order of voter i
        The first value in row i denotes the candidate with the highest preference
        Result is a NumPy array of size (num_voters x num_candidates)
    '''
    return np.array([np.random.permutation(np.arange(1, num_candidates+1)) 
            for _ in range(num_voters)])


def find_winner(profiles, voting_rule):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        In STV, if there is a tie amongst the candidates with minimum plurality score in a round, then eliminate the candidate with the lower index
        For Copeland rule, ties among pairwise competitions lead to half a point for both candidates in their Copeland score

        Return: Index of winning candidate (1-indexed) found using the given voting rule
        If there is a tie amongst the winners, then return the winner with a lower index
    '''

    winner_index = None
    
    # TODO

    num_voters, num_candidates = profiles.shape

    if voting_rule == 'plurality':

        # ith value represents number of times candidate i is the most preferred candidate for a voter
        # Saksham
        count_arr = np.bincount(profiles[:,0])
        # Rathi
        winner_index = np.argmax(count_arr)

    elif voting_rule == 'borda':

        # ith value represents the Borda score of candidate (i+1)
        borda_score_arr = np.zeros(num_candidates)
        for pref in profiles:
            for cand_rank in range(num_candidates):
                candidate = pref[num_candidates - cand_rank - 1]
                borda_score_arr[candidate-1] += cand_rank    

        winner_index = 1 + np.argmax(borda_score_arr)

    elif voting_rule == 'stv':

        # List of candidates removed in each round
        candidates_removed = [0]
        temp_profiles = profiles[:]
        for round in range(num_candidates - 1):
            # Form frequency array for the most preferred candidate of each voter
            count_arr = np.bincount(temp_profiles[:,0])
            count_arr = np.append(count_arr, [0 for _ in range(len(count_arr), num_candidates + 1)])

            # Candidates already removed should no longer be chosen to be removed
            count_arr[candidates_removed] = num_voters + 1
            # Find the candidate with the least frequency (and least index among those)
            worst_candidate = np.argmin(count_arr)
            candidates_removed.append(worst_candidate)
            # Update the preference profiles
            temp_profiles = np.array([np.delete(pref, np.where(pref == worst_candidate)) for pref in temp_profiles])

        winner_index = temp_profiles[0][0]

    elif voting_rule == 'copeland':

        # Cell (i,j) denotes the number of times candidate (i+1) defeats candidate (j+1)
        pairwise_winner_arr = np.zeros((num_candidates, num_candidates))

        for pref in profiles:
            for cand1_rank in range(num_candidates - 1):
                for cand2_rank in range(cand1_rank + 1, num_candidates):
                    winner_cand = pref[cand1_rank]
                    loser_cand = pref[cand2_rank]
                    pairwise_winner_arr[winner_cand - 1][loser_cand - 1] += 1

        copeland_score_arr = np.sum((pairwise_winner_arr > num_voters / 2), axis=1, dtype=np.float64)
        copeland_score_arr += 0.5 * np.sum((pairwise_winner_arr == num_voters / 2), axis=1)
        winner_index = 1 + np.argmax(copeland_score_arr)

    # END TODO

    return winner_index


def find_winner_average_rank(profiles, winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        winner is the index of the winning candidate for some voting rule (1-indexed)

        Return: The average rank of the winning candidate (rank wrt a voter can be from 1 to num_candidates)
    '''

    average_rank = None

    # TODO

    ranks = []
    for pref in profiles:
        ranks.append(1 + np.where(pref == winner)[0])

    average_rank = np.mean(ranks)

    # END TODO

    return average_rank


def check_manipulable(profiles, voting_rule, find_winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        find_winner is a function that takes profiles and voting_rule as input, and gives the winner index as the output
        It is guaranteed that there will be at most 8 candidates if checking manipulability of a voting rule

        Return: Boolean representing whether the voting rule is manipulable for the given preference profiles
    '''

    manipulable = None

    # TODO

    manipulable = False
    num_voters, num_candidates = profiles.shape
    # For each voter, we will iterate over all possible preference orders
    all_prefs = list(itertools.permutations(np.arange(1, num_candidates + 1)))
    winner = find_winner(profiles, voting_rule)

    for voter in range(num_voters):
        pref = profiles[voter]
        temp_profiles = profiles.copy()
        # Change preference order of the current voter to see if it can manipulate
        for new_pref in all_prefs:
            temp_profiles[voter] = new_pref
            new_winner = find_winner(temp_profiles, voting_rule)
            # If new winner is better ranked wrt the old preference, then the voting rule is manipulable
            old_index = np.where(pref == winner)[0][0]
            new_index = np.where(pref == new_winner)[0][0]
            if new_index < old_index:
                manipulable = True
                break

        if manipulable:
            break

    # END TODO

    return manipulable


if __name__ == '__main__':
    np.random.seed(420)

    num_tests = 200
    voting_rules = ['plurality', 'borda', 'stv', 'copeland']

    average_ranks = [[] for _ in range(len(voting_rules))]
    manipulable = [[] for _ in range(len(voting_rules))]
    for _ in tqdm(range(num_tests)):
        # Check average ranks of winner
        num_voters = np.random.choice(np.arange(80, 150))
        num_candidates = np.random.choice(np.arange(10, 80))
        profiles = generate_random_profiles(num_voters, num_candidates)

        for idx, rule in enumerate(voting_rules):
            winner = find_winner(profiles, rule)
            avg_rank = find_winner_average_rank(profiles, winner)
            average_ranks[idx].append(avg_rank / num_candidates)

        # Check if profile is manipulable or not
        num_voters = np.random.choice(np.arange(7, 11))
        num_candidates = np.random.choice(np.arange(3, 7))
        profiles = generate_random_profiles(num_voters, num_candidates)
        
        for idx, rule in enumerate(voting_rules):
            manipulable[idx].append(check_manipulable(profiles, rule, find_winner))


    # Plot average ranks as a histogram
    for idx, rule in enumerate(voting_rules):
        plt.hist(average_ranks[idx], alpha=0.8, label=rule)

    plt.legend()
    plt.xlabel('Fractional average rank of winner')
    plt.ylabel('Frequency')
    plt.savefig('average_ranks.jpg')
    
    # Plot bar chart for fraction of manipulable profiles
    manipulable = np.sum(np.array(manipulable), axis=1)
    manipulable = np.divide(manipulable, num_tests)
    plt.clf()
    plt.bar(voting_rules, manipulable)
    plt.ylabel('Manipulability fraction')
    plt.savefig('manipulable.jpg')