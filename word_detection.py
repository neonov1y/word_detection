import numpy as np
import matplotlib.pyplot as plt
import wave
import os


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

models = {}
accuracy, precision, recall = {}, {}, {}
conf_matrix = {}


def STFT(signal, win, hopSize, F, Fs):
    """
    Input:
        signal 	- signal (numpy vector)
        win		- window (numpy vector in case of window)
                    (scalar in case of Hamming window which specify length of hamming window)
        hopSize - R
        F 		- Number of frequency bins (Even number)
        Fs 		- Sample frequency
    Output:
        STFT (with positive frequencies only)
    """

    # Window generation
    if np.isscalar(win):
        win_len = win
        win = np.hamming(int(win))
    else:
        win_len = len(win)

    # n generation
    if win_len % 2 != 0:
        frame_length = int((win_len - 1) / 2)
    else:
        frame_length = int(win_len / 2)

    N = 1
    while win_len > N:
        N *= 2

    n = np.linspace(-int(N/2), int(N/2) - 1, N, endpoint=True)

    # S initialization
    S = np.array([])

    # Time slots loop
    m = 0
    while 1:
        mR = m * hopSize
        sub_signal = signal[mR: mR + win_len]
        if len(sub_signal) != win_len:
            break
        xm = signal[mR: mR + win_len] * win
        xm_zero_padded = np.zeros(N)
        xm_zero_padded[int(N / 2) - frame_length: int(N / 2) - frame_length + win_len] = xm

        Xm = np.fft.fft(xm_zero_padded, F) / len(xm_zero_padded)
        Xm = Xm[0:int(F / 2)]  # Take only positive frequencies
        freq = np.fft.fftfreq(F, 1 / Fs)
        freq = freq[0:int(F / 2)]  # Take only positive frequencies

        Em = np.exp(-1j * 2 * np.pi * freq * mR / N)
        Sm = np.abs(Xm * Em)

        if m == 0:
            S = np.abs(Sm)
        else:
            S = np.vstack([S, np.abs(Sm)])
        m += 1

    return np.transpose(S)


def get_set():
    dir_path = "data/"
    train_list = list()
    test_list = list()
    for i in range(1, 4):
        i_str = "0" + str(i)
        if i >= 10:
            i_str = str(i)
        files_list_ext = os.listdir(dir_path + i_str + "/")

        for file in files_list_ext:

            if int(file[5]) >= 3 and file[6] != ".":
                test_list.append(dir_path + i_str + "/" + file)
            else:
                train_list.append(dir_path + i_str + "/" + file)

    return train_list, test_list


def stft_clustering(seed=1, plots=False, silence_cancel=False, clustering=False):
    files_list_ext, test_list_ext = get_set()
    print("Clustering:")
    print("\tTrain Set Length:", len(files_list_ext))
    print("\tTest Set Length:", len(test_list_ext))

    padding_len = 20 * 1024
    freq_slots = 256
    time_slots = 20

    Y = np.zeros(shape=(len(files_list_ext)))
    freq_matrix = np.zeros(shape=(len(files_list_ext), freq_slots, time_slots))
    i = 0

    for file in files_list_ext:
        obj = wave.open(file)
        sample_rate = obj.getframerate()
        frames = obj.getnframes()
        audio = obj.readframes(frames)
        audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        audio_len = len(audio_as_np_float32)
        pad_num = max([0, padding_len - audio_len])
        if pad_num != 0:
            pad_audio = np.pad(audio_as_np_float32, (0, pad_num), 'minimum')
        else:
            pad_audio = audio_as_np_float32[0:padding_len]

        freq_stft = STFT(pad_audio, 1024, 1024, 512, sample_rate)
        freq_matrix[i, :, :] = freq_stft

        i += 1

    X = freq_matrix.reshape((len(files_list_ext),freq_slots * time_slots))

    print("\tX dimensions:", X.shape)
    print("\tY dimensions:", Y.shape)

    fact = np.repeat(np.sum(X, axis=1), freq_slots * time_slots).reshape((len(files_list_ext),freq_slots * time_slots)) / (freq_slots * time_slots)
    X = X / fact

    # EDA Clustering
    kmeans = KMeans(n_clusters=2, n_init='auto', random_state=seed)
    kmeans.fit(X)

    predictions = np.array(kmeans.predict(X))

    clust_1_idx = np.where(predictions == 1)[0]
    clust_2_idx = np.where(predictions == 0)[0]

    print("\tFirst 10 elements of elements in cluster 1:")
    for i in clust_1_idx.tolist()[0:10]:
        print("\t\ti:",i,", file:",files_list_ext[i])
        """
        plt.figure()
        plt.title('Average STFT coefficients of Cluster 1')
        plt.imshow(X[i].reshape((freq_slots, time_slots)), cmap="jet")
        plt.colorbar()
        plt.xlabel("m (Time slots)")
        plt.ylabel("k (Frequency)")
        plt.show()
        """
    print("\tFirst 10 elements of elements in cluster 2:")
    for i in clust_2_idx.tolist()[0:10]:
        print("\t\ti:",i,", file:",files_list_ext[i])
        """
        plt.figure()
        plt.title('Average STFT coefficients of Cluster 2')
        plt.imshow(X[i].reshape((freq_slots, time_slots)), cmap="jet")
        plt.colorbar()
        plt.xlabel("m (Time slots)")
        plt.ylabel("k (Frequency)")
        plt.show()
        """

    print("\tNumber of elements in cluster 1:",len(clust_1_idx))
    print("\tNumber of elements in cluster 2:",len(clust_2_idx))

    coeff_mean_cl1 = np.mean(X[clust_1_idx,:], axis=0)
    coeff_mean_cl2 = np.mean(X[clust_2_idx,:], axis=0)

    print("\tMeans Cl1 Shape:",coeff_mean_cl1.shape)
    print("\tMeans Cl2 Shape:",coeff_mean_cl2.shape)

    plt.figure()
    plt.title('First 100 averaged STFT coefficient per cluster')
    plt.plot(range(0, freq_slots * time_slots), coeff_mean_cl1, label="Cluster 1")
    plt.plot(range(0, freq_slots * time_slots), coeff_mean_cl2, label="Cluster 2", alpha=0.7)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.axis((0, 100, 0, 100))
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Average STFT coefficients of Cluster 1')
    plt.imshow(coeff_mean_cl1.reshape((freq_slots, time_slots)), cmap="jet", vmax=60)
    plt.colorbar()
    plt.xlabel("m (Time slots)")
    plt.ylabel("k (Frequency)")
    plt.show()

    plt.figure()
    plt.title('Average STFT coefficients of Cluster 2')
    plt.imshow(coeff_mean_cl2.reshape((freq_slots, time_slots)), cmap="jet", vmax=60)
    plt.colorbar()
    plt.xlabel("m (Time slots)")
    plt.ylabel("k (Frequency)")
    plt.show()


def train_model(seed=1, plots=False, silence_cancel=False):
    files_list_ext, test_list_ext = get_set()
    print("Model train:")
    print("\tTrain Set Length:", len(files_list_ext))
    print("\tTest Set Length:", len(test_list_ext))

    padding_len = 20 * 1024
    freq_slots = 256
    time_slots = 20

    word_len = list()
    not_word_len = list()
    Y = np.zeros(shape=(len(files_list_ext)))
    freq_matrix = np.zeros(shape=(len(files_list_ext), freq_slots, time_slots))
    i = 0

    for file in files_list_ext:
        obj = wave.open(file)
        sample_rate = obj.getframerate()
        frames = obj.getnframes()
        audio = obj.readframes(frames)
        audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        audio_len = len(audio_as_np_float32)
        pad_num = max([0, padding_len - audio_len])
        if pad_num != 0:
            pad_audio = np.pad(audio_as_np_float32, (0, pad_num), 'minimum')
        else:
            pad_audio = audio_as_np_float32[0:padding_len]

        freq_stft = STFT(pad_audio, 1024, 1024, 512, sample_rate)

        if silence_cancel:
            for j in range(0,time_slots):
                s = np.sum(freq_stft[:,j])
                if s > 15:
                    audio_as_np_float32 = audio_as_np_float32[j*1024:]

                    audio_len = len(audio_as_np_float32)
                    pad_num = max([0, padding_len - audio_len])
                    if pad_num != 0:
                        pad_audio = np.pad(audio_as_np_float32, (0, pad_num), 'minimum')
                    else:
                        pad_audio = audio_as_np_float32[0:padding_len]

                    freq_stft = STFT(pad_audio, 1024, 1024, 512, sample_rate)

                    break

        freq_matrix[i, :, :] = freq_stft

        if file[8] == "0":
            Y[i] = 1
            word_len.append(len(audio_as_np_float32))
            """
            plt.figure()
            plt.title('STFT of audio with word "zero"')
            plt.imshow(freq_matrix[i, :, :]/(np.sum(np.sum(freq_matrix))/5120), cmap="jet")
            plt.colorbar()
            plt.xlabel("m (Time slots)")
            plt.ylabel("k (Frequency)")
            plt.show()
            """
        else:
            Y[i] = 0
            not_word_len.append(len(audio_as_np_float32))

        i += 1

    if plots is True:
        # Length analysis
        plt.figure()
        plt.title('Length of record')
        plt.hist(word_len, color="blue", label="Wanted Word Length", rwidth=0.8, bins=10)
        plt.hist(not_word_len, color="orange", label="Unwanted Word Length", rwidth=0.8, alpha=0.75, bins=10)
        plt.xlabel("File index")
        plt.ylabel("Length in Samples")
        plt.legend()
        plt.show()

        print("\tLength statistics:")
        print("\t\tMax length for wanted word:", max(word_len))
        print("\t\tMax length for rest words:", max(not_word_len))

        # Dimensions
        print("\tDimensions:")
        print("\t\tFreq matrix dimension:", freq_matrix.shape)

    X = freq_matrix.reshape((len(files_list_ext),freq_slots * time_slots))

    print("\t\tX dimensions:", X.shape)
    print("\t\tY dimensions:", Y.shape)

    fact = np.repeat(np.sum(X, axis=1), freq_slots * time_slots).reshape((len(files_list_ext),freq_slots * time_slots)) / (freq_slots * time_slots)
    X = X / fact

    # Lasso logistic regression - L1 regularization logistic regression
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # --- To optimize the lambda
    # --- C (float, default=1.0) - Inverse of regularization strength; must be a positive float.
    #   Like in support vector machines, smaller values specify stronger regularization.
    # solver="liblinear"
    model1_l1_LR = LogisticRegression(solver="liblinear", penalty='l1', tol=0.005, random_state=seed)
    distributions = dict(C=np.arange(0.001,2.0,0.02))
    clf = RandomizedSearchCV(model1_l1_LR, distributions, scoring="balanced_accuracy", random_state=seed)
    search = clf.fit(X, Y)
    print("\tCross-validation tuned parameter C (LR):",search.best_params_)

    model1_l1_LR = LogisticRegression(penalty='l1', C=search.best_params_["C"], tol=0.005, solver="liblinear", random_state=seed)
    models["LogisticRegression"] = model1_l1_LR
    model1_l1_LR.fit(X, Y)

    coef_l1_LR = model1_l1_LR.coef_.ravel()
    beta_log_lasso_py = np.array(coef_l1_LR)
    resh_beta_log_lasso_py = beta_log_lasso_py.reshape((freq_slots, time_slots))

    if plots is True:
        plt.figure()
        plt.title('STFT of estimated beta to check which frequencies affect')
        plt.imshow(resh_beta_log_lasso_py != 0, cmap="jet")
        plt.colorbar()
        plt.xlabel("m (Time slots)")
        plt.ylabel("k (Frequency)")
        plt.show()

        plt.figure()
        plt.title('Histogram of beta coefficients')
        plt.hist(beta_log_lasso_py, bins=10)
        plt.xlabel("value")
        plt.ylabel("amount")
        plt.show()

        plt.figure()
        plt.title('Values of beta coefficients')
        plt.bar(range(0,len(beta_log_lasso_py)), beta_log_lasso_py)
        plt.xlabel("index")
        plt.ylabel("value")
        plt.show()

        print("\tNumber of coefficients:",len(beta_log_lasso_py))
        print("\tNumber of non zero coefficients:", sum(beta_log_lasso_py != 0))

    # Support Vector Machines
    models["SVM"] = LinearSVC(penalty='l1', loss='squared_hinge', dual="auto", max_iter=10000, tol=0.005)
    distributions = dict(C=range(10, 50, 5))
    clf = GridSearchCV(models["SVM"], distributions, scoring="balanced_accuracy")
    search = clf.fit(X, Y)
    print("\tCross-validation tuned parameter C (SVM):",search.best_params_)

    models["SVM"] = LinearSVC(penalty='l1', loss='squared_hinge', dual="auto", C=search.best_params_["C"],  tol=0.005)
    models["SVM"].fit(X, Y*2-1)


def test_mult(train_test=True, silence_cancel=False):
    files_list_ext, files_list = get_set()

    print("Testing the model:")
    if train_test is True:
        print("\tTrain set used for testing")
        files_list = files_list_ext
    else:
        print("\tTest set used for testing")

    slot_len = 1024
    pattern_window_size = 20
    freq_slots = 256
    time_slots = 20
    padding_len = 20 * 1024

    Y_test = np.zeros(shape=(len(files_list)))
    X_test = np.zeros(shape=(len(files_list), freq_slots * time_slots))

    i = 0

    for file in files_list:
        obj = wave.open(file)
        sample_rate = obj.getframerate()
        frames = obj.getnframes()
        audio = obj.readframes(frames)
        audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        audio_len = len(audio_as_np_float32)
        pad_num = max([0, padding_len - audio_len])
        if pad_num != 0:
            pad_audio = np.pad(audio_as_np_float32, (0, pad_num), 'minimum')
        else:
            pad_audio = audio_as_np_float32[0:padding_len]

        Si = STFT(pad_audio, slot_len, slot_len, 512, sample_rate)

        if silence_cancel:
            for j in range(0, time_slots):
                s = np.sum(Si[:, j])
                if s > 15:
                    audio_as_np_float32 = audio_as_np_float32[j * 1024:]

                    audio_len = len(audio_as_np_float32)
                    pad_num = max([0, padding_len - audio_len])
                    if pad_num != 0:
                        pad_audio = np.pad(audio_as_np_float32, (0, pad_num), 'minimum')
                    else:
                        pad_audio = audio_as_np_float32[0:padding_len]

                    Si = STFT(pad_audio, 1024, 1024, 512, sample_rate)

                    break

        Y_test[i] = (file[8] == "0")*1
        X_test[i] = Si[:,0:pattern_window_size].reshape((freq_slots * time_slots))

        i+=1

    fact = np.repeat(np.sum(X_test, axis=1), freq_slots * time_slots).reshape((len(files_list),freq_slots * time_slots)) / (freq_slots * time_slots)
    X_test = X_test / fact

    print("\tX dimensions:", X_test.shape)
    print("\tY dimensions:", Y_test.shape)

    for mod in models.keys():
        # Make predictions
        predictions = models[mod].predict(X_test)

        if mod == "SVM":
            Y_test_l = Y_test*2-1
        else:
            Y_test_l = Y_test

        # Calculate metrics
        accuracy[mod] = accuracy_score(predictions, Y_test_l)
        precision[mod] = precision_score(predictions, Y_test_l)
        recall[mod] = recall_score(predictions, Y_test_l)
        conf_matrix[mod] = confusion_matrix(predictions, Y_test_l).ravel()

    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', 'Conf Matrix (TN, FN, FP, TP)'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()
    df_model['Conf Matrix (TN, FN, FP, TP)'] = conf_matrix.values()

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 5,
                           ):
        print(df_model)


def analyse_word_freq():
    print("Frequency analysis:")
    padding_len = 20 * 1024

    obj = wave.open("data/01/0_01_4.wav")
    sample_rate = obj.getframerate()
    frames = obj.getnframes()
    audio = obj.readframes(frames)
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    num_of_steps = 2**17 + 1
    X1 = np.fft.fft(audio_as_np_int16) / len(audio_as_np_int16)
    freq1 = np.fft.fftfreq(len(audio_as_np_int16), 1/sample_rate)

    audio_len = len(audio_as_np_float32)
    pad_num = max([0, padding_len - audio_len])
    if pad_num != 0:
        pad_audio = np.pad(audio_as_np_float32, (0, pad_num), 'minimum')
    else:
        pad_audio = audio_as_np_float32[0:padding_len]
    freq_stft1 = STFT(pad_audio, 1024, 1024, 512, sample_rate)

    obj = wave.open("data/01/1_01_0.wav")
    sample_rate = obj.getframerate()
    frames = obj.getnframes()
    audio = obj.readframes(frames)
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    X2 = np.fft.fft(audio_as_np_int16) / len(audio_as_np_int16)
    freq2 = np.fft.fftfreq(len(audio_as_np_int16), 1/sample_rate)

    obj = wave.open("data/01/2_01_0.wav")
    sample_rate = obj.getframerate()
    frames = obj.getnframes()
    audio = obj.readframes(frames)
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    X3 = np.fft.fft(audio_as_np_int16) / len(audio_as_np_int16)
    freq3 = np.fft.fftfreq(len(audio_as_np_int16), 1/sample_rate)

    obj = wave.open("data/01/3_01_3.wav")
    sample_rate = obj.getframerate()
    frames = obj.getnframes()
    audio = obj.readframes(frames)
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    X4 = np.fft.fft(audio_as_np_int16) / len(audio_as_np_int16)
    freq4 = np.fft.fftfreq(len(audio_as_np_int16), 1/sample_rate)

    # Plot of frequency domain of cut of word "was" and cuts of few other words
    fig, ax = plt.subplots()
    ax.plot(freq1[0:round(len(freq1)/2)], np.abs(X1[0:round(len(X1)/2)]), label="Signal 1 - Zero", linewidth=4)
    ax.plot(freq2[0:round(len(freq2)/2)], np.abs(X2[0:round(len(X2)/2)]), label="Signal 2 - One", linewidth=3)
    ax.plot(freq3[0:round(len(freq3)/2)], np.abs(X3[0:round(len(X3)/2)]), label="Signal 3 - Two", linewidth=2)
    ax.plot(freq4[0:round(len(freq4)/2)], np.abs(X4[0:round(len(X4)/2)]), label="Signal 4 - Three", linewidth=1)
    ax.set(xlabel='frequency (Hz)', ylabel='|X(f)| (DFT of X)', title='DFT of recordings')
    ax.grid()
    ax.legend()
    plt.show()

    plt.figure()
    plt.title("STFT for audio record with word 'zero'")
    plt.imshow(freq_stft1, cmap="jet")
    plt.colorbar()
    plt.xlabel("m (Time slots)")
    plt.ylabel("k (Frequency)")
    plt.show()


def check_the_data():
    files_list_ext, test_list_ext = get_set()
    print("Data check:")
    print("\tTrain Set Length:", len(files_list_ext))
    print("\tTest Set Length:", len(test_list_ext))

    samples_len = list()
    time_len = list()

    for file in files_list_ext + test_list_ext:
        obj = wave.open(file)
        sample_rate = obj.getframerate()
        frames = obj.getnframes()
        audio = obj.readframes(frames)
        audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
        samples_len.append(len(audio_as_np_int16))
        time_len.append(len(audio_as_np_int16)/48000)

    print("\tMaximal length (in number of samples):",max(samples_len))
    print("\tMaximal length (in seconds):", max(samples_len)/48000)
    print("\tMinimal length (in number of samples):",min(samples_len))
    print("\tMinimal length (in seconds):", min(samples_len)/48000)

    # Plot of histogram with length
    fig, ax = plt.subplots()
    ax.hist(time_len, bins=40, color='lightgreen', edgecolor='black')
    ax.set(xlabel='Duration', ylabel='', title='Histogram of recordings length')
    plt.show()


if __name__ == '__main__':
    # Analyze Word Frequencies
    analyse_word_freq()

    # Data Duration Analysis
    check_the_data()

    # EDA clustering
    stft_clustering()

    # Supervised Models
    models = {}
    accuracy, precision, recall = {}, {}, {}
    conf_matrix = {}

    for i in range(1, 2):
        print("Run number:",i)
        train_model(i, False, True)

        # Training set testing
        test_mult(True, True)
        # Tests set testing
        test_mult(False, True)