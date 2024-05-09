import datetime
import csv
import os
import tkinter as tk

now = datetime.datetime.now()
current_date = now.strftime('%Y-%m-%d')

def read_attendance_data():
    attendance_data = []
    with open(f'face_recognition/Face-Recognition-with-YOLO-and-FaceNet/{current_date}.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            attendance_data.append(row)
    return attendance_data

def display_attendance(data):
    index = 1
    for row in data:
        name_label = tk.Label(window, text=row['Name'])
        type_label = tk.Label(window, text=row['Type'])
        date_label = tk.Label(window, text=row['Date'])
        time_label = tk.Label(window, text=row['Time'])
        name_label.grid(row=index, column=0)
        type_label.grid(row=index, column=1)
        date_label.grid(row=index, column=2)
        time_label.grid(row=index, column=3)
        index += 1

    name_header = tk.Label(window, text='Name')
    type_header = tk.Label(window, text='Type')
    date_header = tk.Label(window, text='Date')
    time_header = tk.Label(window, text='Time')
    name_header.grid(row=0, column=0)
    type_header.grid(row=0, column=1)
    date_header.grid(row=0, column=2)
    time_header.grid(row=0, column=3)

window = tk.Tk()
window.minsize(640, 480)
window.title(f'Attendance Logs for {current_date}')

attendance_data = read_attendance_data()

display_attendance(attendance_data)

window.mainloop()
