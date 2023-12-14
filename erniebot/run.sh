

clear


#python APP001/01.cdata.py


#python APP001/02.getEMB.py
#say "I finished th job!"
#python APP001/02.getEMB.py
#say "I finished th job!"
#python APP001/02.getEMB.py
#say "I finished th job!"


python APP001/03.model.py
cd ../prediction_result
rm -rf result6.zip
zip result6.zip bq_corpus.tsv lcqmc.tsv paws-x.tsv

