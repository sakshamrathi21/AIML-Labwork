import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def Gale_Shapley(suitor_prefs, reviewer_prefs) -> Dict[str, str]:
    '''
        Gale-Shapley Algorithm for Stable Matching

        Parameters:

        suitor_prefs: dict - Dictionary of suitor preferences
        reviewer_prefs: dict - Dictionary of reviewer preferences

        Returns:

        matching: dict - Dictionary of suitor matching with reviewer
    '''

    matching = {}

    ## TODO: Implement the Gale-Shapley Algorithm
    suitor_free = list(suitor_prefs.keys())

    while suitor_free:
        suitor = suitor_free.pop(0)
        suitor_choices = suitor_prefs[suitor]

        for reviewer in suitor_choices:
            if reviewer not in matching.values():
                matching[suitor] = reviewer
                break
            else:
                current_suitor = [key for key, value in matching.items() if value == reviewer][0]
                reviewer_choices = reviewer_prefs[reviewer]
                if reviewer_choices.index(suitor) < reviewer_choices.index(current_suitor):
                    suitor_free.append(current_suitor)
                    matching[suitor] = reviewer
                    matching.pop(current_suitor)
                    break
    ## END TODO

    return matching

def avg_suitor_ranking(suitor_prefs: Dict[str, List[str]], matching: Dict[str, str]) -> float:
    '''
        Calculate the average ranking of suitor in the matching
        
        Parameters:
        
        suitor_prefs: dict - Dictionary of suitor preferences
        matching: dict - Dictionary of matching
        
        Returns:
        
        avg_suitor_ranking: float - Average ranking of suitor
    '''

    avg_suitor_ranking = 0

    ## TODO: Implement the average suitor ranking calculation
    for suitor, reviewer in matching.items():
        avg_suitor_ranking += suitor_prefs[suitor].index(reviewer)

    avg_suitor_ranking /= len(matching)
    avg_suitor_ranking += 1
    ## END TODO

    assert type(avg_suitor_ranking) == float

    return avg_suitor_ranking

def avg_reviewer_ranking(reviewer_prefs: Dict[str, List[str]], matching: Dict[str, str]) -> float:
    '''
        Calculate the average ranking of reviewer in the matching
        
        Parameters:
        
        reviewer_prefs: dict - Dictionary of reviewer preferences
        matching: dict - Dictionary of matching
        
        Returns:
        
        avg_reviewer_ranking: float - Average ranking of reviewer
    '''

    avg_reviewer_ranking = 0

    ## TODO: Implement the average reviewer ranking calculation
    for suitor, reviewer in matching.items():
        avg_reviewer_ranking += reviewer_prefs[reviewer].index(suitor)

    avg_reviewer_ranking /= len(matching)
    avg_reviewer_ranking += 1
    ## END TODO

    assert type(avg_reviewer_ranking) == float

    return avg_reviewer_ranking

def get_preferences(file) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    '''
        Get the preferences from the file
        
        Parameters:
        
        file: file - File containing the preferences
        
        Returns:
        
        suitor_prefs: dict - Dictionary of suitor preferences
        reviewer_prefs: dict - Dictionary of reviewer preferences
    '''
    suitor_prefs = {}
    reviewer_prefs = {}

    for line in file:
        if line[0].islower():
            reviewer, prefs = line.strip().split(' : ')
            reviewer_prefs[reviewer] = prefs.split()

        else:
            suitor, prefs = line.strip().split(' : ')
            suitor_prefs[suitor] = prefs.split()
        
    return suitor_prefs, reviewer_prefs


if __name__ == '__main__':

    avg_suitor_ranking_list = []
    avg_reviewer_ranking_list = []

    for i in range(100):
        with open('data/data_'+str(i)+'.txt', 'r') as f:
            suitor_prefs, reviewer_prefs = get_preferences(f)

            # suitor_prefs = {
            #     'A': ['a', 'b', 'c'],
            #     'B': ['c', 'b', 'a'],
            #     'C': ['c', 'a', 'b']
            # }

            # reviewer_prefs = {
            #     'a': ['A', 'C', 'B'],
            #     'b': ['B', 'A', 'C'],
            #     'c': ['B', 'A', 'C']
            # }

            matching = Gale_Shapley(suitor_prefs, reviewer_prefs)

            avg_suitor_ranking_list.append(avg_suitor_ranking(suitor_prefs, matching))
            avg_reviewer_ranking_list.append(avg_reviewer_ranking(reviewer_prefs, matching))

    plt.hist(avg_suitor_ranking_list, bins=10, label='Suitor', alpha=0.5, color='r')
    plt.hist(avg_reviewer_ranking_list, bins=10, label='Reviewer', alpha=0.5, color='g')

    plt.xlabel('Average Ranking')
    plt.ylabel('Frequency')

    plt.legend()
    plt.savefig('q2.png')


    

