import os
import datetime

version = "#7"
scoreL = """
0.5782
0.9918
0.8448
0.8306
0.7951
0.7100
"""
scoreL = [float(i) for i in scoreL.split("\n") if i]
score = (scoreL[0]+scoreL[1]+scoreL[4])/3
print(scoreL, round(score, 4))


with open("README.md", "r") as f:
    base = f.readlines()

fUP = lambda new, old: '+' if new > old else '-'


with open("README.md", "w") as f:
    for i in base:
        
        #
        if "TASK-" in i:
            old = float(i[114:].replace("+","").replace("-","").replace(" ||\n", ""))
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
            old = float(i[13:].replace("+","").replace("-","").replace(" ||\n", ""))
            new = (scoreL[0]+scoreL[1]+scoreL[4])/3
            
            i = f"{i[:13]} {new:.4f} |{fUP(new, old)}|\n"
        
        #
        if "Updated" in i:
            i = f"{i[:10]} {str(datetime.datetime.now())}.\n"
        
        f.write(i)

os.system(f"""git add * && git commit -m "{version} {score:.4f}" && git push""")
        


