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

    if voting_rule == 'plurality':
        # print(profiles)
        first_preferences_of_each_voter = profiles[:, 0]
        num_voters, num_candidates = np.shape(profiles)
        votes_for_each_candidate = np.zeros(num_candidates+1)
        for candidate in first_preferences_of_each_voter:
            votes_for_each_candidate[candidate]+=1
        max_votes = 0

        for i in range(num_candidates+1):
            if (votes_for_each_candidate[i] > max_votes):
                winner_index = i 
                max_votes = votes_for_each_candidate[i]

    elif voting_rule == 'borda':
        num_voters, num_candidates = np.shape(profiles)
        points = np.zeros(num_candidates+1)
        for i in range(num_voters):
            for j in range(num_candidates):
                points[profiles[i][j]] += (num_candidates - j - 1)
        winner_index = np.argmax(points)
        
    elif voting_rule == 'stv':
        num_voters, num_candidates = np.shape(profiles)
        candidate_indices = np.arange(1, num_candidates + 1)
        remaining_candidates = set(candidate_indices)
        while len(remaining_candidates) > 1:
            votes = np.zeros(num_candidates)
            for row in profiles:
                for candidate in row:
                    if candidate in remaining_candidates:
                        votes[candidate - 1] += 1
                        break
            min_votes = min(votes[candidate - 1] for candidate in remaining_candidates)
            for candidate in list(remaining_candidates):
                if votes[candidate - 1] == min_votes:
                    remaining_candidates.remove(candidate)
                    break
        winner_index = remaining_candidates.pop()

    elif voting_rule == 'copeland':
        num_voters, num_candidates = np.shape(profiles)
        copeland_scores = np.zeros(num_candidates)

        reverse_profiles = np.zeros_like(profiles)
        for i in range(num_voters):
            for j in range(num_candidates):
                reverse_profiles[i][profiles[i][j]-1] = j

        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                scorei=0
                scorej=0
                for k in range(num_voters):
                    if(reverse_profiles[k][i]>reverse_profiles[k][j]):
                        scorej+=1
                    if(reverse_profiles[k][i]<reverse_profiles[k][j]):
                        scorei+=1
                if(scorei < scorej):
                    copeland_scores[j] +=1
                elif(scorei > scorej):
                    copeland_scores[i] +=1
                else:
                    copeland_scores[j] +=0.5
                    copeland_scores[i] +=0.5

        winner_index = np.argmax(copeland_scores) + 1
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

    ranks = np.where(profiles == winner)[1] + 1
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

    num_voters, num_candidates = profiles.shape
    og_winner = find_winner(profiles, voting_rule)
    perms=itertools.permutations(profiles[0])
    
    for voter in range(num_voters):
        for permutation in perms:
            newprofiles = np.copy(profiles)
            newprofiles[voter] = permutation
            new_winner = find_winner(newprofiles, voting_rule)
            # print(new_winner, profiles[voter], np.where(profiles[voter] == new_winner)[0][0])
            if new_winner != og_winner and np.where(profiles[voter] == new_winner)[0][0] < np.where(profiles[voter] == og_winner)[0][0]:
                return True
        # profiles[voter] = old_preferences

    return False

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
        num_voters = np.random.choice(np.arange(10, 20))
        num_candidates = np.random.choice(np.arange(4, 8))
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