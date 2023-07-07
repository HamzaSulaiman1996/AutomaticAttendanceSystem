import cv2
from pathlib import Path
import numpy as np
import pickle
import face_recognition


def main():
    Reg = Registration()
    Reg.start()
    print("Thankyou for registering...\nClosing...")


class Registration():
    base = Path('.').absolute()

    def __init__(self):

        '''Root Folder'''

        self.reg_fold = self.base / 'Registry'
        self.reg_fold.mkdir(parents=True, exist_ok=True)

    def start(self):
        while True:
            self.c_name = None
            self.opt = input(
                'Register new Course: R\nAdd students to existing course: A\nSave Data: S\nExit Program: E\n-> ')
            if self.opt == 'R':
                self.__makereg()

            elif self.opt == 'A':
                self.__append()

            elif self.opt == 'S':
                self.__save()

            elif self.opt == 'E':
                return False

            else:
                print('Please select a valid option!\n->')

    def __makereg(self):  ##   Adding New Courses
        self.c_name = input('Enter the course name:\n->')

        print('Enter starting time of class\n')
        self.h = input('Enter hour:\n->')
        self.m = input('Enter minute:\n->')
        self.s = input('Enter second:\n->')
        self.time = self.h + ':' + self.m + ':' + self.s

        try:
            self.c_fol = self.reg_fold / self.c_name
            self.c_fol.mkdir(parents=True)
            print('Course added successfully!')
            quer = input(f'Add students to course?: Y/N\n->')
            if quer == 'Y':
                self.__append()
            elif quer == 'N':
                return None

        except Exception as e:  ## Prompt User that Course already Exists if
            print(f'Course name {self.c_name} already exists!\n')  ## Existing Course Name entered

    def __append(self):

        if self.c_name is not None:  ## not None is only True when the user already added a new course in above method __makereg()
            self.s_name = input('Enter the student name:\n->')
            self.s_id = input('Enter the student ID:\n->')
            self.__capture()

        else:
            ## this is executed if c_name is None
            ## if c_name is None meaning user wants to add entry inside some existing course

            if len(list(self.reg_fold.iterdir())) == 0:  ## check whether there is any Course in directory
                print('No course available\n')
                return None

            while True:

                print('list of courses:')  ##printing available list of courses
                for courses in self.reg_fold.iterdir():
                    print((str(courses).split('\\'))[-1])

                self.c_name = input('Which course do you want to register:\n->')

                self.c_fol = self.reg_fold / self.c_name

                if self.c_fol.exists() == False:  ## prompt user if invalid course name entered

                    print('*************\nPlease Select from available courses!\n*************\n')
                    continue
                else:
                    break

            self.s_name = input('Enter the student name:\n->')
            self.s_id = input('Enter the student ID:\n->')

            self.__capture()

    def __capture(self):
        print('Enter c to capture image')

        fol_path = self.c_fol / self.s_name
        fol_path.mkdir(parents=True, exist_ok=True)

        ## simple haarcascades are used to detect face in the camera
        ## will save only the image inside bounding box 
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        cap = cv2.VideoCapture(0)
        while True:
            _, img = cap.read()
            save_img = img.copy()

            '''Simple haarcascade from cv2 for face detection'''

            gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_scale, 1.05, 5)

            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
                width = x + w
                height = y + h + 20
                cv2.rectangle(img, (x, y - 20), (width, height), (0, 255, 0), 2)

            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                # save_img = save_img[y-20:height,x:width]
                # print(f'{fol_path / self.s_id}')
                # cv2.imwrite(f'{fol_path / self.s_id}.png',save_img)
                cv2.imwrite(f'{fol_path / self.s_id}.jpg', save_img)

                break

        cap.release()
        cv2.destroyAllWindows()
        print('Picture Taken Successfully')
        print('----------------------------')

    def __train(self):
        self.temp = []  ## storing the names of students whose faces will get encoded for creation of Names.txt file later

        print('Available Courses:')
        print('******************')

        for num_c in self.reg_fold.iterdir():  ## show user the available list of courses to choose
            c = str(num_c).split('\\')[-1]
            print(c)

        self.c_fol = input('Choose course to program:\n->')

        name = self.__load_temp()  ## load names from previous Names.txt file

        while True:

            c_dir = self.reg_fold / self.c_fol  ##simple loop to execute if user enters invalid course name
            if not (c_dir.exists()):
                print('*************\nPlease Select from available courses!\n*************\n')
                continue
            else:
                break

        self.known_names = []  ## will be used to store names of students
        self.known_faces = []  ## will be used to store encoded values of faces

        ##        for i, st_files in enumerate(c_dir.iterdir()):  ## iterate over every course directory
        for st_files in c_dir.iterdir():
            not_found = 0

            f_name = (str(st_files).split('\\')[-1])  ##fetch the course name from string of course directory

            if f_name in name:  ## if encoding already done for faces in course folder, skip the folder
                print(f'{f_name} already saved. Saving others...')
                continue

            self.temp.append(f_name)  ##if new course folder, append name in temp list
            for sts in st_files.iterdir():  ##iterate over every image file inside the course

                try:
                    '''use of face_recognition library for encoding faces in the image'''
                    image = face_recognition.load_image_file(str(sts))
                    encoding = face_recognition.face_encodings(image)[0]
                    self.known_faces.append(encoding)
                    self.known_names.append(f_name)

                except Exception as e:
                    not_found += 1  ##simple count to tell how many faces not detected in individual folder

            print(f'{not_found} Face(s) not found in folder:{f_name}')
        print('Training done!')

        return None

    def __load_temp(self):
        names = []

        path = self.base / self.c_fol
        temp_path = path / 'Names.txt'
        if Path(temp_path).exists() == False:  ## this will only be executed if program runs brand new, in this case the Names.txt is not created yet
            return []
        else:
            with Path(temp_path).open('r') as f:
                for m in f:
                    names.append(m.split("\n")[0])  ## add all the names in the list

            return names

    def __save_temp(self):

        temp_file_1 = self.course_dir / 'Names.txt'
        if temp_file_1.exists():  ## True when the temp file has already been created before

            with Path(temp_file_1).open('a') as f:
                for names in self.temp:
                    ##                    courses = f.write(f'{names}\n')
                    f.write(f'{names}\n')
        else:
            with Path(temp_file_1).open('x') as f:  ##creation of the temp.txt file for the first time
                for names in self.temp:
                    ##                    courses = f.write(f'{names}\n')
                    f.write(f'{names}\n')

        temp_file_2 = self.course_dir / 'course_time.txt'
        if not (temp_file_2.exists()):
            with Path(temp_file_2).open('x') as f:
                f.write(self.time)

    def __save(self):
        self.__train()

        self.course_dir = self.base / self.c_fol
        self.course_dir.mkdir(parents=True, exist_ok=True)  ##Creation of the Course Directory

        self.name_file = self.course_dir / 'Names.data'  ##Creation of Binary Name file
        self.face_file = self.course_dir / 'Face_enc.data'  ##Creation of Binary Face encoded file

        if self.name_file.exists():  ##If True, append on exisiting data

            with Path(self.name_file).open('rb') as f:
                temp1 = pickle.load(f)
            [temp1.append(i) for i in self.known_names]

            with Path(self.name_file).open('wb') as f:
                pickle.dump(temp1, f)


        else:
            with Path(self.name_file).open('xb') as f:
                pickle.dump(self.known_names, f)

        if self.face_file.exists():  ##If True, append on exisiting data
            with Path(self.face_file).open('rb') as f:
                temp1 = pickle.load(f)
            [temp1.append(i) for i in self.known_faces]

            with Path(self.face_file).open('wb') as f:
                pickle.dump(temp1, f)


        else:
            with Path(self.face_file).open('xb') as f:
                pickle.dump(self.known_faces, f)

        self.__save_temp()  ##save Names.txt file

        print('Data Saved..')


if __name__ == '__main__':
    main()
