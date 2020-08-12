Step1: Unzip the folder and navigate to the folder using command prompt
cmd> cd /path/to/folder/

Step 2: Create virtual environment using anaconda
cmd> conda create -n assignment python==3.7 in your command prompt

Step 3: Activate conda environment 
cmd> conda activate assignment

Step 4: Install the required packages
cmd> pip install -r requirements.txt

Step 5: Place the data to be tested in the /data/ folder

Step 6: open the config file located in /src/ folder
And change the projects_path, models_path and data_path as per your system
Change the test_file_name: to the filename the model needs to be tested upon

Step 7: Go to command prompt and type:
cmd> python test.py

This will calculate the root mean squared error, accuray of the model as per the desired condition in the assignment document
Also a predictions.csv file will be generated in the /data/ folder
