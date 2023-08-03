import glob
import json
import os 
jsonfiles = glob.glob("/mnt/parscratch/users/acq22zm/babylm-challenge/process-data/tc/*.json")



def get_score(file, metric):
    '''Get the score of a given metric'''

    data_name = os.path.basename(os.path.normpath(file))
    data_name = data_name.replace(".json", ".train.conllu")

    # storing_scores = {}

    with open(file, "r") as f:

        contents = json.load(f)


        # print(data_name)
        score = contents[data_name][metric]["value"]


    return data_name, score

# metrics = 

dependents_per_word = {}
average_dependency_distance = {}
rarity = {}
lexical_density = {}
dispersion = {}
dispersion_ent = {}

for x in jsonfiles:

    data_name, score = get_score(x, "dependents per word")
    # dependents_per_word = get_score(x, "dependents per word")

    dependents_per_word[data_name] = score


    data_name, score = get_score(x, "average dependency distance")
    average_dependency_distance[data_name] = score

    data_name, score = get_score(x, "rarity")
    rarity[data_name] = score

    data_name, score = get_score(x, "lexical density")
    lexical_density[data_name] = score

    data_name, score = get_score(x, "Gini-based dispersion (disjoint windows)")
    dispersion[data_name] = score

    # entropy (disjoint windows)

    # data_name, score = get_score(x, r"entropy")
    # dispersion_ent[data_name] = score
    

dependents_per_word = dict(sorted(dependents_per_word.items(), key=lambda item: item[1]))
average_dependency_distance = dict(sorted(average_dependency_distance.items(), key=lambda item: item[1]))
rarity = dict(sorted(rarity.items(), key=lambda item: item[1]))
lexical_density = dict(sorted(lexical_density.items(), key=lambda item: item[1]))
dispersion = dict(sorted(dispersion.items(), key=lambda item: item[1]))
# dispersion_ent = dict(sorted(dispersion_ent.items(), key=lambda item: item[1]))


print(f"DPW = {dependents_per_word}")
print(f"ADD = {average_dependency_distance}")
print(f"R = {rarity}")
print(f"LD = {lexical_density}")
print(f"D = {dispersion}")
# print(f"D_ent = {dispersion_ent}")

# DPW ={'aochildes.train.conllu': 0.7596166936417444, 
#       'open_subtitles.train.conllu': 0.8066823161718212, 
#       'bnc_spoken.train.conllu': 0.8187492125640581, 
#       'switchboard.train.conllu': 0.8237719481786246, 
#       'qed.train.conllu': 0.8594396020247024, 
#       'cbt.train.conllu': 0.8729895462662653, 
#       'simple_wikipedia.train.conllu': 0.8966519363588512, 
#       'gutenberg.train.conllu': 0.9023866762810108, 
#       'wikipedia.train.conllu': 0.9403610009889145, 
#       'children_stories.train.conllu': 0.9405928556829873}

# ADD = {'aochildes.train.conllu': 1.8336698653216792, 
#        'open_subtitles.train.conllu': 2.140132370349912, 
#        'bnc_spoken.train.conllu': 2.4219043382060312, 
#        'switchboard.train.conllu': 2.5630895879571303, 
#        'qed.train.conllu': 2.587316524265169, 
#        'simple_wikipedia.train.conllu': 2.800130327002362, 
#        'gutenberg.train.conllu': 3.024535025639056, 
#        'cbt.train.conllu': 3.174739212279776, 
#        'wikipedia.train.conllu': 3.1929998295553506, 
#        'children_stories.train.conllu': 3.4308745511072503}

# R = {'bnc_spoken.train.conllu': 0.18889306086775745, 
#      'switchboard.train.conllu': 0.1938331263745306, 
#      'qed.train.conllu': 0.20800843631899563, 
#      'open_subtitles.train.conllu': 0.22538734086174447, 
#      'children_stories.train.conllu': 0.23926693189296874, 
#      'simple_wikipedia.train.conllu': 0.24927411288474294, 
#      'aochildes.train.conllu': 0.26065548899626234, 
#      'cbt.train.conllu': 0.26557995306738186, 
#      'wikipedia.train.conllu': 0.29110696851968365, 
#      'gutenberg.train.conllu': 0.31106530535163956}



# LD = {'simple_wikipedia.train.conllu': 0.36240059889560744, 
#       'wikipedia.train.conllu': 0.36520803951935094, 
#       'children_stories.train.conllu': 0.4115740797227882, 
#       'gutenberg.train.conllu': 0.41322062940197846, 
#       'cbt.train.conllu': 0.4132487579505882, 
#       'open_subtitles.train.conllu': 0.41963524361106674, 
#       'switchboard.train.conllu': 0.442312909864542, 
#       'qed.train.conllu': 0.4449209527272381, 
#       'bnc_spoken.train.conllu': 0.4618339569182658, 
#       'aochildes.train.conllu': 0.47304403423918717}

# D = {'simple_wikipedia.train.conllu': 0.3085284446160619, 
#      'aochildes.train.conllu': 0.31445264891944974, 
#      'wikipedia.train.conllu': 0.33736966246923883, 
#      'qed.train.conllu': 0.36561260497282877, 
#      'bnc_spoken.train.conllu': 0.3714121676791458, 
#      'switchboard.train.conllu': 0.38381788237938974, 
#      'open_subtitles.train.conllu': 0.38974238145388496, 
#      'gutenberg.train.conllu': 0.419101206142359, 
#      'children_stories.train.conllu': 0.4278684915892836, 
#      'cbt.train.conllu': 0.4435469239239907}