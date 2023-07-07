import cv2
from pathlib import Path
import numpy as np
import face_recognition
import pickle
import csv
from datetime import datetime,date


def main():

    start = Recognition()
    start.display() 


class Recognition():

    def __init__(self):
        
        self.root = Path('.').absolute()
        
        path_1 = self.root / 'Face_enc.data'
        path_2 = self.root / 'Names.data'


        with Path(path_1).open('rb') as f:
            self.known_faces = pickle.load(f)   ##Load encoded face data

        with Path(path_2).open('rb') as f:
            self.known_names = pickle.load(f)   ##Load encoded name data
            

        self.header = ['Name','Late','Ontime']  ##Headers of Late_Ontime csv file

        self.Lo_path = self.root / 'Late_Ontime.csv'  ##Path to the csv file
        
        if  not(self.Lo_path.exists()):     ## if file doesnot exisits, create new file and add headers 
            
            with Path(self.Lo_path).open('x') as f:
                writer = csv.DictWriter(f,fieldnames=self.header)
                writer.writeheader()



        self.per_day_path = self.root / 'Per_Day.csv'   ##Path to the csv file


        if  not(self.per_day_path.exists()):  ## if file doesnot exisits, create new file and add headers 
                    
            with Path(self.per_day_path).open('x') as f:
                
                header = ['Name']
                writer = csv.DictWriter(f,fieldnames=header)
                writer.writeheader()

        
        with Path(self.per_day_path).open('r') as f: ##add a new header with current date everytime program starts
            
            reader = csv.DictReader(f)
            self.append_date = reader.fieldnames + [self.get_date()]   


    def display(self):
        
        self.marked = [] ##list of added students in the database
        cap = cv2.VideoCapture(0)
        while True:
            _,img = cap.read()


            ''' face recognition algorithm '''
            locations = face_recognition.face_locations(img)
            encodings = face_recognition.face_encodings(img,locations)
            if len(encodings)!=0:

                for location,encoding in zip(locations,encodings):
                    compare = face_recognition.compare_faces(self.known_faces,encoding,0.6) ##gives a boolean list with True being the index of encoded face data
           
                    name = 'Unknown'
                    if np.any(compare): ## condition to check whether a face is recognised or not
                        name = self.known_names[np.argmax(compare)] ## get the name of the encoded face
                        if name not in self.marked:  ##to prevent adding same names again
                            self.marked.append(name) ##new name is appended on the self.marked list
                            self.attendance(name)


                    top_left = (location[3],location[0])
                    bottom_right = (location[1],location[2])
                    color = [0,255,0]
                    cv2.rectangle(img,top_left,bottom_right,color,3)
                    cv2.putText(img,name,(location[3]+10,location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,(200,200,200),2)

            cv2.imshow('',img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
      

        cap.release()
        cv2.destroyAllWindows()


    def attendance(self,name):

        self.per_day(name)
        self.late_ontime(name)

        
    def per_day(self,name):

        with Path(self.per_day_path).open('r',newline='') as f:
            NAME_FOUND = False
            reader = csv.DictReader(f)
            lists = list(reader)  ##convert to list for ease
            for row in lists:

                if (name in row.values()):  ##if name already available in csv file, then only update the row
                    row.update( {self.get_date():self.get_time()} )
                    NAME_FOUND = True
                    break


            if not(NAME_FOUND): ##if new entry, then add a new row 
                self.new_entry_per_day(name,lists)


        with Path(self.per_day_path).open('w',newline='') as f: ##save the csv file

            writer = csv.DictWriter(f,fieldnames=self.append_date)
            writer.writeheader()
            writer.writerows(lists)


    def late_ontime(self,name):

        with Path(self.Lo_path).open('r',newline='') as f:
            NAME_FOUND = False
            reader = csv.DictReader(f)
            lists = list(reader)
            for row in lists:

        
                if ( name in row.values() ): ##if name already available in csv file, then only update the row
                    self.status(row)
                    NAME_FOUND = True
                    break

    
            if not(NAME_FOUND): ##if new entry, then add a new row
                self.new_entry_late_ontime(name,lists)



        with Path(self.Lo_path).open('w',newline='') as f: ##save the csv file
            writer = csv.DictWriter(f,fieldnames=self.header)
            writer.writeheader()
            writer.writerows(lists)
        
        

    def time_check(self):  ##check starting class time with the student arrival time
        
        now = self.get_time()
        
        with Path('course_time.txt').open('r') as f:
            c_time = f.read()

        if now > c_time:
            return [1,0] ##late
        else:
            return [0,1] ##ontime



    def status(self,row):  ## for updating student row in Late_Ontime csv file
    
        s = self.time_check()

        temp = {'Late':int(row['Late'])+s[0],'Ontime':int(row['Ontime'])+s[1]}
        row.update(temp)


    def new_entry_late_ontime(self,name,lists): ## adding new entry in Late_Ontime csv file

        s = self.time_check()
        f_status = {'Name':name,'Late':s[0],'Ontime':s[1]}
        lists.append(f_status)



    def new_entry_per_day(self,name,lists):  ## adding new entry in Per_Day csv file
        f = {'Name':name,self.get_date():self.get_time()}
        lists.append(f)


    def get_date(self): ##get current data
##        return '10-07-21'
        return date.today().strftime("%d-%m-%y")

    def get_time(self):  ##get current time
        return datetime.now().strftime("%H:%M:%S")
         

if __name__ == '__main__':
    
    main()
