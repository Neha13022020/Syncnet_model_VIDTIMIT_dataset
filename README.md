# Syncnet_model_VIDTIMIT_dataset

There is a lot of advancement in video manipulation techniques. And it's a lot easier to create tampered videos, which can fool human eyes. Such content leads to fake news or misinformation.

So in this project, I tried to detect whether the video is tampered or not, or you can say, real or fake? I referred to research paper - "https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf" throughout my project. They focused on determining the audio-video synchronization between mouth motion and speech in the video. They used audio-video synchronization for TV broadcasting. It's really a nice research paper, they developed a language-independent and speaker-independent solution to the lip-sync problem, without labeled data. 

For the modeling and processing functions I referred "https://github.com/voletiv/syncnet-in-keras". Here I used the VidTIMIT dataset (http://conradsanderson.id.au/vidtimit/) for my project. 
