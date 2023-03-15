import os
import datetime

os.system("clear")

p1 = "#7"
p2 = "Epoch:51"
scoreL = """
0.5881
0.9994
0.8376
0.8372
0.7948
0.7095
"""
test = "ACC: 0.90180 F1: 0.89135"

scoreL = [float(i) for i in scoreL.split("\n") if i]
score = (scoreL[0]/0.6365 + scoreL[1]/0.9998 + scoreL[4]/0.9428)/3
print(scoreL, round(score, 4), "\n")


with open("README.md", "r") as f:
    base = f.readlines()
    
#
fOLD = lambda x: float(x.replace("+","").replace("-","").replace("=","").replace(" ||\n", ""))
fUP = lambda new, old: '+' if new > old else ('=' if new == old else '-')


with open("README.md", "w") as f:
    for i in base:
        
        #
        if "TASK-" in i:
            old = fOLD(i[114:])
            if "TASK-3" in i:
                new = scoreL[0]
                
            if "TASK-5" in i:
                new = scoreL[1]
        
            #
            if "TASK-6" in i and "ACC" in i:
                new = scoreL[4]
            
            if "TASK-6" in i and "lcqmc" in i:
                new = scoreL[2]
            
            if "TASK-6" in i and "bq_corpus" in i:
                new = scoreL[3]
            
            if "TASK-6" in i and "paws-x" in i:
                new = scoreL[5]
            
            i = f"{i[:114]} {new:.4f} |{fUP(new, old)}|\n"
        
        #
        if "Total" in i:
            old = fOLD(i[13:])
            new = score
            
            i = f"{i[:13]} {new:.4f} |{fUP(new, old)}|\n"
        
        #
        if "Updated" in i:
            i = f"{i[:10]} {str(datetime.datetime.now())}, Test {test}.\n"
        
        f.write(i)

#
os.system(f"""git add * && git commit -m "{p1} {p2} /{score:.4f} ~" && git push""")
        


