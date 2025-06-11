from tkinter import *
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytesseract
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Initialize Window
main = Tk()
main.title("Vehicle Pattern Recognition using Machine & Deep Learning to Predict Car Model")
main.geometry("1300x850")
main.configure(bg="#f2f2f2")

# Styling
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=('Segoe UI', 11), padding=6)
style.configure("TLabel", font=('Segoe UI', 12, 'bold'))
style.configure("TFrame", background="#f2f2f2")

# Globals
filename = None
accuracy = []
classifier = None
names = ['AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012',
         'Acura TL Type-S 2008', 'Acura TSX Sedan 2012']

# Menu Bar
menubar = Menu(main)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Upload Dataset", command=lambda: uploadDataset())
filemenu.add_separator()
filemenu.add_command(label="Exit", command=lambda: main.destroy())
menubar.add_cascade(label="File", menu=filemenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Vehicle Recognition App using ML/DL\nDeveloped with Tkinter"))
menubar.add_cascade(label="Help", menu=helpmenu)

main.config(menu=menubar)

# Title Label
ttk.Label(main, text='Vehicle Pattern Recognition using Machine & Deep Learning to Predict Car Model',
          font=('Segoe UI', 16, 'bold'), background="#f2f2f2", anchor='center').pack(pady=20)

# Layout Frames
container = Frame(main, bg="#f2f2f2")
container.pack(padx=20, pady=10, fill=BOTH, expand=True)

left_frame = ttk.Frame(container, width=400)
left_frame.pack(side=LEFT, fill=Y)

right_frame = ttk.Frame(container)
right_frame.pack(side=RIGHT, fill=BOTH, expand=True)


def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, 'Dataset loaded\n')
    X = np.load("model/X.txt.npy")
    img = X[20].reshape(64, 64, 3)
    cv2.imshow('Sample Image', cv2.resize(img, (250, 250)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def linearKNN():
    
    accuracy.clear()
    text.delete('1.0', END)
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    XX = np.reshape(X, (X.shape[0], -1))
    XX = PCA(n_components=180).fit_transform(XX)

    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

    cls = LogisticRegression(max_iter=1000)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test, predict) * 100
    accuracy.append(acc)
    text.insert(END, 'Linear Regression Accuracy : {:.2f}%\n'.format(acc))

    cls = KNeighborsClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test, predict) * 100
    accuracy.append(acc)
    text.insert(END, 'KNN Accuracy : {:.2f}%\n'.format(acc))

    plt.bar(['Linear Regression', 'KNN'], accuracy)
    plt.title('Linear Regression & KNN Accuracy')
    plt.show()

def SVMCNN():
    global classifier
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    XX = np.reshape(X, (X.shape[0], -1))
    XX = PCA(n_components=180).fit_transform(XX)

    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test, predict) * 100
    accuracy.append(acc)
    text.insert(END, 'SVM Accuracy : {:.2f}%\n'.format(acc))

    Y1 = to_categorical(Y)
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(units=256, activation='relu'))
    cnn.add(Dense(units=5, activation='softmax'))
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = cnn.fit(X, Y1, batch_size=16, epochs=10, shuffle=True, verbose=2)
    acc = history.history['accuracy'][-1] * 100
    accuracy.append(acc)
    classifier = cnn

    text.insert(END, 'CNN Accuracy : {:.2f}%\n'.format(acc))

    plt.bar(['SVM', 'CNN'], accuracy[-2:])
    plt.title('SVM & CNN Accuracy')
    plt.show()

def KNNSVM():
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    XX = np.reshape(X, (X.shape[0], -1))
    XX = PCA(n_components=180).fit_transform(XX)

    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

    cls = KNeighborsClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test, predict) * 100
    accuracy.append(acc)
    text.insert(END, 'KNN Accuracy : {:.2f}%\n'.format(acc))

    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test, predict) * 100
    accuracy.append(acc)
    text.insert(END, 'SVM Accuracy : {:.2f}%\n'.format(acc))

    plt.bar(['KNN Inference', 'SVM Inference'], accuracy[-2:])
    plt.title('KNN & SVM Inference Accuracy')
    plt.show()

def KNNCNN():
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    XX = np.reshape(X, (X.shape[0], -1))
    XX = PCA(n_components=180).fit_transform(XX)

    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

    cls = KNeighborsClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test, predict) * 100
    accuracy.append(acc)
    text.insert(END, 'KNN Accuracy : {:.2f}%\n'.format(acc))

    Y1 = to_categorical(Y)
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(units=128, activation='relu'))
    cnn.add(Dense(units=5, activation='softmax'))
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = cnn.fit(X, Y1, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
    acc = history.history['accuracy'][-1] * 100
    accuracy.append(acc)

    text.insert(END, 'CNN Accuracy : {:.2f}%\n'.format(acc))
    plt.bar(['KNN Inference', 'CNN Inference'], [accuracy[6], accuracy[7]])
    plt.title('KNN & CNN Inference Accuracy')
    plt.show()

def predict():
    global classifier
    if classifier is None:
        messagebox.showerror("Error", "Please run the CNN model first!")
        return

    filename = filedialog.askopenfilename(initialdir="testImages")
    if not filename:
        return

    image = cv2.imread(filename)
    img = cv2.resize(image, (64, 64)).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    preds = classifier.predict(img)
    predict_idx = np.argmax(preds)
    result_text = 'Car Model Predicted as: ' + names[predict_idx]

    messagebox.showinfo("Prediction Result", result_text)

    img_disp = cv2.cvtColor(cv2.resize(image, (400, 300)), cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_disp)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    top = Toplevel()
    top.title(result_text)
    label = Label(top, image=img_tk)
    label.image = img_tk
    label.pack()

def detectNumberPlate():
    try:
        file = filedialog.askopenfilename(initialdir="testImages")
        if not file:
            return

        image = cv2.imread(file)
        if image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 30, 200)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        plate_found = False

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(c)
                cropped = image[y:y + h, x:x + w]
                text_raw = pytesseract.image_to_string(cropped, config='--psm 11')
                text_cleaned = text_raw.strip().upper()

                if not text_cleaned:
                    continue

                plate_found = True
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                cv2.imshow("Detected Plate", cropped)

                region_map = {
                    'TS': 'Telangana', 
                    'AP': 'Andhra Pradesh', 
                    'TN': 'Tamil Nadu', 
                    'KL': 'Kerala',
                    'KA': 'Karnataka', 
                    'MH': 'Maharashtra', 
                    'GJ': 'Gujarat', 
                    'RJ': 'Rajasthan',
                    'UP': 'Uttar Pradesh', 
                    'HR': 'Haryana', 
                    'PB': 'Punjab', 
                    'JK': 'Jammu and Kashmir',
                    'HP': 'Himachal Pradesh', 
                    'UK': 'Uttarakhand', 
                    'OR': 'Odisha', 
                    'WB': 'West Bengal',
                    'BR': 'Bihar', 
                    'CG': 'Chhattisgarh', 
                    'MP': 'Madhya Pradesh'
                }

                plate_code = text_cleaned[:2]
                region = region_map.get(plate_code, "Unknown Region")

                messagebox.showinfo("Number Plate", f"Detected Number: {text_cleaned}")
                messagebox.showinfo("Region", f"Region Detected: {region}")
                break

        if not plate_found:
            messagebox.showwarning("Warning", "No Number Plate Found")

        cv2.imshow("Plate Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Error", f"Error in detecting number plate:\n{str(e)}")

def graph():
    bars = ['Linear Regression', 'KNN', 'SVM', 'CNN', 'KNN Inference', 'SVM Inference', 'KNN Inference 2', 'CNN Inference 2']
    y_pos = np.arange(len(accuracy))
    plt.bar(y_pos, accuracy)
    plt.xticks(y_pos, bars[:len(accuracy)], rotation=45)
    plt.title('All Algorithms Accuracy Comparison')
    plt.tight_layout()
    plt.show()

def close():
    main.destroy()



# Button Definitions
buttons = [
    ("Upload Cars Dataset", uploadDataset),
    ("Run Linear Regression & KNN", linearKNN),
    ("Run SVM & CNN", SVMCNN),
    ("Run KNN & SVM", KNNSVM),
    ("Run KNN & CNN", KNNCNN),
    ("Prediction Model", predict),
    ("Accuracy Comparison Graph", graph),
    ("Detect Number Plate", detectNumberPlate)
]

for text_val, command_func in buttons:
    ttk.Button(left_frame, text=text_val, command=command_func, width=35).pack(pady=7)

# Output Text Box
text = Text(right_frame, height=35, width=85, font=('Segoe UI', 11))
text.pack(pady=10)

main.mainloop()
