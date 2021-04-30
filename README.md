# Caption This Presentation

This project was constructed in accordance with the requirements outlined in Galvanize's Data Science Immersive curriculum for Capstone 3 projects. I was a part of the Remote Oct2020-Jan2021 Cohort(RFT5) and completed my course of study on January 15, 2021.

## The Project

My original idea was to build an end-to-end speech recognition software system that could take in live, streaming audio and rapidly create a transcription of the audio. The goal was to be able to use my product to live caption myself giving the final presentation at the end. I am proud to say that I succeeded in achieving this goal, although I had to utilize transfer learning rather than be able to build it end-to-end from scratch. But let's get into the details!

### Step 1: Download and Install libraries

First step is to pip-install the requirements.txt file at the top level of this repo. This is best done in a virtual environment to avoid dependency issues, but it can also be done in your standard base environment.

``` python
sudo pip install -r requirements.txt
```

One additional thing to note is that portaudio is required, but this often has dependency difficulties, so I recommend to conda install portaudio rather than pip and let the solver try to sort out dependencies for your environment.

### Step 2: Repo's Needed

As mentioned, this project relies on transfer learning, but it also uses several open source github repos to achieve it's end goal. Therefore, I mention one repo that is required, and then offer some optional to improve performance (which will be detailed later). The other repo I mention is where I obtained my pretrained model, as well as garnered insight into much of this project.https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant

(Required)
* ctcdecode (https://github.com/parlance/ctcdecode)

(Optional)
* Mimic Recoding Studio by MycroftAI (https://github.com/MycroftAI/mimic-recording-studio)
* KenLM (https://github.com/kpu/kenlm)

(Credit)
* Michael Phi's A Hacker's Voice Recognition Assistant (https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant)

The directories for these repos are included in this repo, but for better use it is advised to clone the repos directly to your system outside of the directory where you clone this repo. I will come back and polish up this repo one day soon, but for now it contains all relevant libraries and code used to develop the project! Note that several substantial changes are made to files in Michael Phi's transfer learned model because I encountered numerous dependency errors and run-time errors when trying to get the model running.

### Step 3 (optional): Record Your Voice

Using Mimic Recording Studio, I fine-tuned the pretrained model on about an hour of my own voice (which with augmentation was really much more than that) to improve performance. I am certain that if you capture more than an hour of your own voice, you will see even better performance than my own implementation.

Once you have your own voice recordings, use the script in neuralnet/scripts called mimic_create_jsons.py (courtesy of LearnedVector) to easily create an acceptable json file for training. If you use something else to create your wav files besides Mimic Recording Studio, jsonifyaudio.py in the src folder (my own code) will help you get the data into acceptable format.

### Step 4: Training

Use train.py in the speechrecognition folder for training. Currently, it accepts audio up to about 15 seconds in length, but if there are clips that are longer the training will still run and it will just skip those files(along with the appropriate labels). I used the pretrained model provided on Michael Phi's github for his own ASR project, but you are welcome to tweak and train your own model. Be warned, the training will be intensive, as my P3 EC2 on AWS ran at 97% CPU usage with 16 vCPU's!

![Loss_Graph](https://github.com/Rasputin1127/Caption_This_Presentation/blob/main/src/images/val_loss.jpg)

### Step 5: Optimizing Model for Production

Again, big thanks to Michael Phi for his code that turns his pretrained model into a frozen, optimized torch object. Use optimize_graph.py in the neuralnet folder inside of the speechrecognition directory to accomplish this after training/fine-tuning your own model!

### Step 6 (optional): Language Model

If you cloned the kenLM repo, in my data folder there is a folder called language_model_texts that contains a text document with around 8 million words, all compiled from various tv shows, movies, and wikipedia. There is also an arpa file that can be pointed to directly when you run the presentation, or you can train your own kenLM arpa file and point it to that. Either way, adding in this language model component significantly helps with the end product!

(Some helpful code to use if you have trouble parsing the kenLM repo even after getting it built)

``` python
bin/lmplz -o 5 <text >text.arpa
```

Use this block from the 'build' directory of your kenLM repo after building it with cmake, and replace text with the .txt file you want to train the model on. Text.arpa can be named whatever-you-want.arpa, and this is what you point to when demoing your model.

### Step 7: Demo

Within the speechrecognition folder is a demo folder that contains the flask app. I have implemented Carousel through Bootstrap to populate my slideshow into the flask app, and you can do the same by just replacing the images appropriately in the image folder and in the code. Using --args, point to your optimized model and to your language model arpa file (if you have one) and you will be good to go for your own demonstration!

```python
python demo.py --model_file path/to/file --ken_lm_file path/to/lm/file
```

Here is the link to a YouTube video of my final presenation, feel free to comment on it to let me know what you think of it and to suggest any improvements!

https://youtu.be/ARugFR710Hw

Also feel free to reach out on LinkedIn: /kyle-boerstler
