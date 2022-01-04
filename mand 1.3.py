#Loading Libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk,Image
import glob
import os
import webbrowser
import PyPDF2
resumeDataSet = pd.read_csv('F:\\GUI course\\Machine learning of application\\UpdatedResumeDataSet.csv' ,encoding='utf-8')
#Data Preprocessing
def cleanResume(resumeText):
    resumeText = re.sub('httpS+s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText) 
    resumeText = re.sub('s+', ' ', resumeText)  # remove extra whitespace
    return resumeText
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1000)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

#Model Building
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

def get_result_cv():
    path='%s.pdf'%(user_name.get())
    # creating a pdf file object 
    pdfFileObj = open(path, 'rb') 
    # creating a pdf reader object 
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
    # printing number of pages in pdf file 
    print(pdfReader.numPages) 
    # creating a page object 
    pageObj = pdfReader.getPage(0)
    try:
       page_2  = pdfReader.getPage(1)
       cv  =  pageObj.extractText()+page_2.extractText()
    # extracting text from page
    except:
       cv  =  pageObj.extractText()  
    # closing the pdf file object 
    pdfFileObj.close()
    t=pd.DataFrame([[cv],[resumeDataSet.iloc[0][1]],[resumeDataSet.iloc[1][1]],[resumeDataSet.iloc[2][1]],[resumeDataSet.iloc[3][1]],[resumeDataSet.iloc[4][1]],[resumeDataSet.iloc[5][1]],[resumeDataSet.iloc[6][1]],[resumeDataSet.iloc[7][1]],[resumeDataSet.iloc[8][1]],[resumeDataSet.iloc[9][1]],[resumeDataSet.iloc[10][1]],[resumeDataSet.iloc[11][1]],[resumeDataSet.iloc[12][1]],[resumeDataSet.iloc[13][1]],[resumeDataSet.iloc[15][1]],[resumeDataSet.iloc[16][1]],[resumeDataSet.iloc[17][1]],[resumeDataSet.iloc[18][1]],[resumeDataSet.iloc[19][1]],[resumeDataSet.iloc[20][1]]],columns=['A'])
    t['cleaned_resume'] = t.A.apply(lambda x: cleanResume(x))
    cv_clean_1 = t['cleaned_resume'].values
    word_vectorizer_1 = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1000,
    lowercase=True   )
    word_vectorizer_1.fit(cv_clean_1)
    WordFeatures_1 = word_vectorizer_1.transform(cv_clean_1)
    pre=clf.predict(WordFeatures_1)
    list_cv=["Advocate",
        "Arts",
        "Automation Testing"
       ,"Blockchain"
       ,"Business Analyst"
       ,"Civil Engineer"
       ,"Data Science"
       ,"Database"
       ,"DevOps Engineer"
       ,"DotNet Developer"
       ,"ETL Developer"
       ,"Electrical Engineering"
       ,"HR"
       ,"Hadoop"
       ,"Health and fitness"
       ,"Java Developer"
       ,"Mechanical Engineer"
       ,"Network Security Engineer"
       ,"Operations Manager"
       ,"PMO"
       ,"Python Developer"
       ,"SAP Developer"
       ,"Sales"
       ,"Testing"
       ,"Web Designing"]
    w=list_cv[pre[0]]
    print(w)
    label_18=ttk.Label(awad,text=f"Your CV lead you to work as {w}")
    label_18.pack()
    return w
    
def get_draw():
    label_10.forget()
    label_11.forget()
    label_12.forget()
    label_13.forget()
    label_14.forget()
    label_15.forget()
    label_16.pack(side="left")
    label_17.pack(side="left")


def searchininternet():
    os.startfile("C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")
    #chrome=webbrowser.get("C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s")
    #chrome.open_new_tab("www.google.com")

def openexcel():
    os.startfile("C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Microsoft Office\Microsoft Excel 2010.lnk")


def openpdf():
    x='%s.pdf'%(user_name.get())
    try:
       os.startfile(x)
    except:
       label_3.forget() 
       label_5.pack()


def cvphoto():
        x='F:/GUI course/Projects/New folder\\%s.png'%(user_name.get())
        if x in list:
            xindex= list.index(x)
            label_2["image"]=image_list[xindex]
            label_2.pack()
        else:
            label_5.forget()
            label_3.pack()
            
            
        
def clearcv():
    if(label_2):
          label_2.pack_forget()
    if (label_3):      
          label_3.pack_forget()
    
list=glob.glob('F:/GUI course/Projects/New folder/*.png')
image_list=[]


awad=tk.Tk()
root=tk.Tk()
awad.geometry("1300x600")
awad.title("machine learning")
root.geometry("600x800")
root.title("Mando App")
style=ttk.Style(root)
style.theme_use('classic')
root=tk.Toplevel()
#root.iconbitmap(r"F:\GUI course\Projects\1.png")

top_frame=tk.Frame(root)
top_frame.pack(fill="both")

label_1=ttk.Label(top_frame,text='Name: ',padding=(0,10))
label_1.pack(side="left")

#inaialize entery
user_name=tk.StringVar(value="")
name_enter=ttk.Entry(top_frame,width=100,textvariable=user_name)
name_enter.pack(side="left")
name_enter.focus() #focus consle in it

#Add frame 2
bottom_frame=tk.Frame(root)
bottom_frame.pack(fill="both")

#Add frame 3
middel_frame=tk.Frame(root)
middel_frame.pack(fill="both")


#Add button to do action(window,text=,function to do it command=)
greet_a=ttk.Button(bottom_frame,text='Get CV',command=cvphoto)
greet_a.pack(side="left",fill="both",expand="True")

#Add button
clear=ttk.Button(bottom_frame,text='Clear CV',command=clearcv)
clear.pack(side="left",fill="both",expand="True")

#Add pdf button
greet_a=ttk.Button(bottom_frame,text='Open PDF',command=openpdf)
greet_a.pack(side="left",fill="both",expand="True")


#Add open excel
greet_a=ttk.Button(bottom_frame,text='Open Excel',command=openexcel)
greet_a.pack(side="left",fill="both",expand="True")


#Add google button
greet_a=ttk.Button(bottom_frame,text='Open Browser',command=searchininternet)
greet_a.pack(side="left",fill="both",expand="True")



#Add button for quit
quit_window=ttk.Button(middel_frame,text='Exit',command=root.destroy)
quit_window.pack(side="left",fill="both",expand="True")


for i in list:
    image=Image.open(i).resize((500,500))
    photo=ImageTk.PhotoImage(image)
    image_list.append(photo)

label_2=ttk.Label(root,image=image_list[0])
label_2.pack()

label_4=ttk.Label(root,text=f"We have {len(image_list)-1} CV",background='red')
label_4.pack()

label_3=ttk.Label(root,text="Image of CV not exist") 
label_5= ttk.Label(root,text="PDF of CV not exist")   

#Run the window but the code stops here until the loop is end.

label_10=ttk.Label(awad,text="Resume Screening with Natural Language Processing",font="20",background="blue",foreground="white")
frame_5=ttk.Frame(awad)
label_11=ttk.Label(frame_5,text="Why do we need Resume Screening?")
label_12=ttk.Label(frame_5,text="For each recruitment, companies take out online ads, referrals and go through them manually.\nCompanies often submit thousands of resumes for every posting.When companies collect resumes through online advertisements, they categorize those resumes according to their requirements.\nAfter collecting resumes, companies close advertisements and online applying portals.\nThen they send the collected resumes to the Hiring Team(s).\nIt becomes very difficult for the hiring teams to read the resume and select the resume according to the requirement, there is no problem if there are one or two resumes but it is very difficult\n to go through 1000’s resumes and select the best one.To solve this problem,\n today in this article we will read and screen the resume using machine learning with Python so that we can complete days of work in few minutes")
label_10.pack()
frame_6=ttk.Frame(awad)
frame_5.pack(fill="both")
label_11.pack(side="left")
label_12.pack(side="left")
frame_6.pack(fill="both")
label_13=ttk.Label(frame_6,text="What is Resume Screening?")
label_13.pack(side="left")
label_14=ttk.Label(frame_6,text="Choosing the right people for the job is the biggest responsibility of every business\n since choosing the right set of people can accelerate business growth exponentially We will discuss here an example of such a business, which we know as\n the IT department. We know that the IT department falls short of growing markets\nDue to many big projects with big companies, their team does not have time to read resumes and choose the best resume according to their requirements.To solve this type of problem, the company always chooses a \nthird party whose job is to make the resume as per the requirement. These companies are known by the name of Hiring Service Organization. It’s all about the information resume screen.\nThe work of selecting the best talent, assignments, online coding contests among many others is also known as resume screen.\nDue to lack of time, big companies do not have enough time to open resumes, due to which they have to take the help of any other company. For which they have to pay money. Which is a very serious problem.\nTo solve this problem, the company wants to start the work of the resume screen itself by using a machine learning algorithm.")
label_14.pack(side="left")
label_15=ttk.Label(awad,text="We get our data from Kaggle",background="red")
label_15.pack()
image_1=Image.open("F:\GUI course\Machine learning of application\catogry_dist.png").resize((700,700))
photo_1=ImageTk.PhotoImage(image_1)
image_2=Image.open("F:\GUI course\Machine learning of application\job_catogry.png").resize((700,700))
photo_2=ImageTk.PhotoImage(image_2)
label_16=ttk.Label(awad,image=photo_1)
label_17=ttk.Label(awad,image=photo_2)
button_draw=ttk.Button(awad,text="Get categorical drawing",command=get_draw)
button_draw.pack()
button_draw_1=ttk.Button(awad,text="try your CV",command=get_result_cv)
button_draw_1.pack()
root.mainloop() 
awad.mainloop()


