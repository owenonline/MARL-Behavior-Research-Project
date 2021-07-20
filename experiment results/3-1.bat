cd "C:\Users\owenb\Desktop\experiment results\python\regression_training" 
"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\python.exe" correlation_assessment_agent3_1.py
move agent3-1.h5 "C:\Users\owenb\Desktop\experiment results\trained regression models"
cd "C:\Users\owenb\Desktop\experiment results\python\posttrain runs"
"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\python.exe" agent3_1_posttrain_runs.py
cd "C:\Users\owenb\Desktop\experiment results\python\pretrain runs"
"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\python.exe" agent3_1_pretrain_runs.py