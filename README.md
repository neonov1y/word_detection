Here I am sharing the code from a project on specific word detection in audio with speech. The purpose was to investigate the methods and create a proof-of-concept application, but not to achieve the best performance.

During the project, I investigated Statistical Learning methods such as Logistic Regression and SVM (with L1 norm regularization) for specific word detection in audio recordings with speech. I also used clustering for model improvement. STFT (Short Time Fourier Transform) coefficients of the audio signal were used as features.

The code concentrated in one file and include functions which implements: STFT transform, analysis of data set, models training and testing ana analysis using clustering.

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

**stft_clustering(seed=1)** - 
```
Input:

Output:

```

**train_model(seed=1, plots=False, silence_cancel=False)** -
```
Input:

Output:

```

**test_mult(train_test=True, silence_cancel=False)** -
```
Input:

Output:

```

**analyse_word_freq()** -
```
Input:

Output:

```

**check_the_data()** -
```
Input:

Output:

```
