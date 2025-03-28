Step 1: Execute `run_quick_test.bat`  
        to detect and install the required libraries in the virtual environment.
        

Step 2: Execute `run_project_on_windows.bat`  
        to install the required libraries indicated in `requirements.txt`  
        and automatically run `test_for_gpu.py`.

Step 3: Run your script via VS Code's integrated terminal or any terminal.


On Windows:

python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

On Linux/Mac:

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
