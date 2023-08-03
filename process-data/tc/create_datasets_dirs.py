from os import system

DPW = {'aochildes.train': 0.7596166936417444, 'open_subtitles.train': 0.8066823161718212, 'bnc_spoken.train': 0.8187492125640581, 'switchboard.train': 0.8237719481786246, 'qed.train': 0.8594396020247024, 'cbt.train': 0.8729895462662653, 'simple_wikipedia.train': 0.8966519363588512, 'gutenberg.train': 0.9023866762810108, 'wikipedia.train': 0.9403610009889145, 'children_stories.train': 0.9405928556829873}
ADD = {'aochildes.train': 1.8336698653216792, 'open_subtitles.train': 2.140132370349912, 'bnc_spoken.train': 2.4219043382060312, 'switchboard.train': 2.5630895879571303, 'qed.train': 2.587316524265169, 'simple_wikipedia.train': 2.800130327002362, 'gutenberg.train': 3.024535025639056, 'cbt.train': 3.174739212279776, 'wikipedia.train': 3.1929998295553506, 'children_stories.train': 3.4308745511072503}
R = {'bnc_spoken.train': 0.18889306086775745, 'switchboard.train': 0.1938331263745306, 'qed.train': 0.20800843631899563, 'open_subtitles.train': 0.22538734086174447, 'children_stories.train': 0.23926693189296874, 'simple_wikipedia.train': 0.24927411288474294, 'aochildes.train': 0.26065548899626234, 'cbt.train': 0.26557995306738186, 'wikipedia.train': 0.29110696851968365, 'gutenberg.train': 0.31106530535163956}
LD = {'simple_wikipedia.train': 0.36240059889560744, 'wikipedia.train': 0.36520803951935094, 'children_stories.train': 0.4115740797227882, 'gutenberg.train': 0.41322062940197846, 'cbt.train': 0.4132487579505882, 'open_subtitles.train': 0.41963524361106674, 'switchboard.train': 0.442312909864542, 'qed.train': 0.4449209527272381, 'bnc_spoken.train': 0.4618339569182658, 'aochildes.train': 0.47304403423918717}
D = {'simple_wikipedia.train': 0.3085284446160619, 'aochildes.train': 0.31445264891944974, 'wikipedia.train': 0.33736966246923883, 'qed.train': 0.36561260497282877, 'bnc_spoken.train': 0.3714121676791458, 'switchboard.train': 0.38381788237938974, 'open_subtitles.train': 0.38974238145388496, 'gutenberg.train': 0.419101206142359, 'children_stories.train': 0.4278684915892836, 'cbt.train': 0.4435469239239907}


system("mkdir /mnt/parscratch/users/acq22zm/babylm-challenge/data-dpw")
system("mkdir /mnt/parscratch/users/acq22zm/babylm-challenge/data-add")
system("mkdir /mnt/parscratch/users/acq22zm/babylm-challenge/data-r")
system("mkdir /mnt/parscratch/users/acq22zm/babylm-challenge/data-ld")
system("mkdir /mnt/parscratch/users/acq22zm/babylm-challenge/data-d")

    