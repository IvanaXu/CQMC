
clear

if [ $1 -eq 1 ]
then
    echo 1
    python3 APP001/01.cdata.py
fi

if [ $1 -eq 2 ]
then
    echo 2
    python3 APP001/02.getEMB.py
fi

if [ $1 -eq 3 ]
then
    echo 3
    python3 APP001/03.model.py
    cd ../prediction_result
    rm -rf result6.zip
    zip result6.zip bq_corpus.tsv lcqmc.tsv paws-x.tsv
fi

if [ $1 -eq 4 ]
then
    echo 4
    cd ../
    python3 README.md.py
fi
