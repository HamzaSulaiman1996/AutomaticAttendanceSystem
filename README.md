# AutomaticAttendanceSystem
### This project demonstrates the use of face recognition for automatic attendance such as in a Classroom environment.

- Install dependencies
```
pip install -r requirements.txt
```
> Tutorial on how to install ``face-recognition`` properly can be found [here](https://www.geeksforgeeks.org/how-to-install-face-recognition-in-python-on-windows/)

> Note: This project uses the ``face-recognition`` library which is only compatible with Python version 3.8 and 3.7

- The ``Registration.py`` script accomplishes the following tasks:
  - Course Creation
  - Adding Student Database
  - Creating face embeddings for recognition

- In ``Recognition.py``, the algorithm will detect and recognize every face from the camera feed and create two csv files namely “Per_Day.csv” and “Late_Ontime.csv” respectively. These CSV files will contain information on every student who attends the class.
  - Per_Day.csv: In this csv file, the arrival time of every student in class will be available column-wise.
  - Late_Ontime.csv: In this csv file, the number of times a student has arrived late or early will be available.
    
> Note: This script should be placed inside each Course directory which will be created after the course has been created by the ``Registration.py`` script.
