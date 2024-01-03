
clear

if [ $1 -eq 1 ]
then
    echo 1
    /data/soft/python3/bin/python APP001/01.cdata.py
fi

if [ $1 -eq 2 ]
then
    echo 2
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
    /data/soft/python3/bin/python APP001/02.getEMB.py && sleep 60
fi

if [ $1 -eq 3 ]
then
    echo 3
    /data/soft/python3/bin/python APP001/03.model.py
    cd ../prediction_result
    rm -rf result6.zip
    zip result6.zip bq_corpus.tsv lcqmc.tsv paws-x.tsv
fi
