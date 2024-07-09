Here I am sharing the code from a project on specific word detection in audio with speech. The purpose was to investigate the methods and create a proof-of-concept application, but not to achieve the best performance.

The blog post can be found here:  

During the project, I investigated Statistical Learning methods such as Logistic Regression and SVM (with L1 norm regularization) for specific word detection in audio recordings with speech. I also used clustering for model improvement. STFT (Short Time Fourier Transform) coefficients of the audio signal were used as features.

The code concentrated in one file and include functions which implements: STFT transform, analysis of data set, models training and testing ana analysis using clustering. Also, provided the dataset which used for training and testing and it can be found in "data" directory.

See below the functions discription.

# Functions discription:

**STFT(signal, win, hopSize, F, Fs)** - implementation of STFT transform
```
Input:
   signal 	- signal (numpy vector)
   win		- window (numpy vector in case of window) 
     (scalar in case of Hamming window which specify length of hamming window) 
   hopSize - R
   F 		- Number of frequency bins (Even number) 
   Fs 		- Sample frequency
Output:
  STFT (with positive frequencies only)
```

**stft_clustering(seed=1)** - k-mean clustering with two clusters using STFT coefficients, plots averaged STFT coefficients per cluster for analysis
```
Input:
   seed - seed used in function
Output:
```

**train_model(seed=1, plots=False, silence_cancel=False)** - training of Logistic Regression and SVM models
```
Input:
   seed - seed used in function
   plot - flag which used to make plots of beta coefficients for analysis
   silence_cancel - flag used to set preproceccing option for silence removal in start of the audio
Output:
```

**test_mult(train_test=True, silence_cancel=False)** - testing of Logistic Regression and SVM models (plots performance information)
```
Input:
   train_test - boolean flag which defined if to use training set or testing set
   silence_cancel - - flag used to set preproceccing option for silence removal in start of the audio (should be the same value as used in train_model function)
Output:
```

**analyse_word_freq()** - DFT analysis of audio recordings with 4 different words 
```
Input:
Output:
```

**check_the_data()** - statistical information about duration of audio records in dataset
```
Input:
Output:
```
