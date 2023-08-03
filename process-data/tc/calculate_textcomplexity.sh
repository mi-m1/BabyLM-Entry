#!/bin/bash

python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu aochildes.train.conllu; >> aochildes.json
python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu bnc_spoken.train.conllu; >> bnc_spoken.json
python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu cbt.train.conllu; >> cbt.json
python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu children_stories.train.conllu; >> children_stories.json
python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu gutenberg.train.conllu; >> gutenberg.json
python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu open_subtitles.train.conllu; >> open_subtitles.json
python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu qed.train.conllu; >> qed.json
python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu simple_wikipedia.train.conllu; >> simple_wikipedia.json
python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu switchboard.train.conllu; >> switchboard.json
python ../../textcomplexity/bin/txtcomplexity --preset all --lang en -i conllu wikipedia.train.conllu; >> wikipedia.json