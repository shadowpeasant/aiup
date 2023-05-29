import numpy as np
import librosa
import IPython.display as ipd
import soundfile as sf
from omegaconf import DictConfig
import pytorch_lightning as pl
import os 
import librosa.display
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
import json
import nemo
import nemo.collections.asr as nemo_asr
from ipywebrtc import CameraStream, ImageRecorder, AudioRecorder
from moviepy.editor import *

from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
from tqdm import tqdm

import sys
IN_COLAB = 'google.colab' in sys.modules

# Samples captured through the microphone from COlab.
#
colab_sample_rate = None
colab_audio = None

# These are functions that are imported into the Notebook
#
nemo_model_for_speech_recognition = None
nemo_model_params = None

# For the audio recorder.
#
audio_recorder = None

# Displays a HTML control to playback the audio
#
def playback_audio(audio_file):
    samples, sample_rate = librosa.load(audio_file)
    return ipd.Audio(audio_file, rate=sample_rate)


# This function displays the audio waveform of an audio file
#
def display_audio_waveform(audio_file):
    samples, sample_rate = librosa.load(audio_file)
    
    # Plot our example audio file's waveform
    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Waveform of Audio Example')
    plt.ylabel('Amplitude')

    _ = librosa.display.waveshow(samples)
    
# This function displays the audio spectrogram of an audio file
#
def display_audio_spectrogram(audio_file):
    samples, sample_rate = librosa.load(audio_file)
    
    # Get spectrogram using Librosa's Short-Time Fourier Transform (stft)
    spec = np.abs(librosa.stft(samples))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)  # Decibels

    # Use log scale to view frequencies
    librosa.display.specshow(spec_db, y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Audio Spectrogram');
    
# This function displays the audio mel-scaled spectrogram of an audio file
#
def display_audio_mel_spectrogram(audio_file):
    samples, sample_rate = librosa.load(audio_file)
    
    mel_spec = librosa.feature.melspectrogram(samples, sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    librosa.display.specshow(
        mel_spec_db, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title('Mel Spectrogram');    

    
# Converts all SPH files in a folder and its sub-folders into WAV and 
# saves a copy of that WAV file alongside the SPH file.
#
def convert_sph_files_to_wav(folder):

    filepathlist = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            if name.endswith(".sph"):
                filepathlist.append(os.path.join(root, name))

    for filepath in tqdm(filepathlist):
        samples, sample_rate = librosa.load(filepath, sr=16000)
        sf.write(filepath.replace(".sph", ".wav"), samples, sample_rate)
    print ("Complete.")
                

# This function loads the Nemo model parameters from the given
# YAML file.
#
def load_parameters_from_config(config_path):
    global nemo_model_params
    
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        nemo_model_params = yaml.load(f)
    return nemo_model_params

# Process and get all unique words
#
def get_all_unique_words(transcripts_path):
    unique_words = {}
    with open(transcripts_path, 'r') as fin:
        for line in fin:
            transcript = line[: line.find('(')-1].lower()
            transcript = transcript.replace('<s>', '').replace('</s>', '')
            transcript = transcript.strip()
            words = transcript.split(' ')
            
            for word in words:
                unique_words[word] = 1
    return ' '.join(sorted(unique_words.keys()))
    

# Function to build a manifest for the AN4 dataset
#
def build_manifest_for_an4_dataset(transcripts_path, wav_path, manifest_path):
    with open(transcripts_path, 'r') as fin:
        with open(manifest_path, 'w') as fout:
            lines = []
            for line in fin:
                lines.append(line)

            for line in tqdm(lines):
                # Lines look like this:
                # <s> transcript </s> (fileID)
                transcript = line[: line.find('(')-1].lower()
                transcript = transcript.replace('<s>', '').replace('</s>', '')
                transcript = transcript.strip()

                file_id = line[line.find('(')+1 : -2]  # e.g. "cen4-fash-b"
                audio_path = os.path.join(
                    wav_path,
                    file_id[file_id.find('-')+1 : file_id.rfind('-')],
                    file_id + '.wav')

                duration =  float(os.path.getsize(audio_path)) / 2 / 16000 #librosa.core.get_duration(filename=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                }
                json.dump(metadata, fout)
                fout.write('\n')
                
# Create a new model for speech recognition
#
def create_speech_recognition_nemo_model(config_path, train_manifest="", test_manifest=""):
    global nemo_model_for_speech_recognition, nemo_model_params
    
    # Load the model parameters from the path
    #
    nemo_model_params = load_parameters_from_config(config_path)
    nemo_model_params['model']['train_ds']['manifest_filepath'] = train_manifest
    nemo_model_params['model']['validation_ds']['manifest_filepath'] = test_manifest

    # Create our ASR model
    #
    nemo_model_for_speech_recognition = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(nemo_model_params['model']))
    
    
# Create a new model for speech recognition
#
def create_speech_recognition_nemo_model_pretrained(config_path, model_name):
    global nemo_model_for_speech_recognition, nemo_model_params
    
    # Load the model parameters from the path
    #
    nemo_model_params = load_parameters_from_config(config_path)

    # Create our ASR model
    #
    nemo_model_for_speech_recognition = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    

def set_train_test_manifest(train_manifest, test_manifest):
    global nemo_model_for_speech_recognition, nemo_model_params
    nemo_model_params['model']['train_ds']['manifest_filepath'] = train_manifest
    nemo_model_params['model']['train_ds']['batch_size'] = 8
    nemo_model_params['model']['validation_ds']['manifest_filepath'] = test_manifest
    nemo_model_params['model']['validation_ds']['batch_size'] = 8
    nemo_model_for_speech_recognition.setup_training_data(train_data_config=nemo_model_params['model']['train_ds'])
    nemo_model_for_speech_recognition.setup_validation_data(val_data_config=nemo_model_params['model']['validation_ds'])
    
def set_learning_rate(learning_rate):
    global nemo_model_for_speech_recognition, nemo_model_params
    nemo_model_params['model']['optim']['lr'] = learning_rate
    nemo_model_for_speech_recognition.setup_optimization(optim_config=DictConfig(nemo_model_params['model']['optim']))
    
    
def train_speech_recognition_nemo_model(batch_size=16, gpus=1, max_epochs=50):
    global nemo_model_for_speech_recognition, nemo_model_params

    # Set the batch-size for the training and test data
    nemo_model_params['model']['train_ds']['batch_size'] = batch_size
    nemo_model_for_speech_recognition.setup_training_data(train_data_config=nemo_model_params['model']['train_ds'])

    nemo_model_params['model']['validation_ds']['batch_size'] = batch_size
    nemo_model_for_speech_recognition.setup_training_data(train_data_config=nemo_model_params['model']['validation_ds'])

    nemo_model_for_speech_recognition.cuda()
    
    # This creates a PyTorch trainer
    #
    trainer = pl.Trainer(gpus=gpus, max_epochs=max_epochs)

    # Train our model
    #
    trainer.fit(nemo_model_for_speech_recognition)    
    
def save_speech_recognition_nemo_model(file):
    global nemo_model_for_speech_recognition, nemo_model_params
    json.dump(nemo_model_params, open(file + ".params", "w"))
    nemo_model_for_speech_recognition.save_to(file)
    
    
def load_speech_recognition_nemo_model(file):
    global nemo_model_for_speech_recognition, nemo_model_params
    nemo_model_params = json.load(open(file + ".params", "r"))
    nemo_model_for_speech_recognition = nemo_asr.models.EncDecCTCModel.restore_from(file) 
              

def get_sr():
  return nemo_model_params, nemo_model_for_speech_recognition

# This method computes the Word Error Rate given a 
# model and the validation data set set up in the 
# params["model"]["validation_ds"] field.
#
def compute_wer(dataset_manifest, wbs=None):

    global nemo_model_for_speech_recognition, nemo_model_params
    
    # Bigger batch-size = bigger throughput
    nemo_model_params['model']['validation_ds']['manifest_filepath'] = dataset_manifest
    nemo_model_params['model']['validation_ds']['batch_size'] = 8

    # Setup the test data loader and make sure the model is on GPU
    nemo_model_for_speech_recognition.eval()
    nemo_model_for_speech_recognition.setup_test_data(test_data_config=nemo_model_params['model']['validation_ds'])
    nemo_model_for_speech_recognition.cuda()

    # We will be computing Word Error Rate (WER) metric between our hypothesis and predictions.
    # WER is computed as numerator/denominator.
    # We'll gather all the test batches' numerators and denominators.
    wer_nums = []
    wer_denoms = []

    # Loop over all test batches.
    # Iterating over the model's `test_dataloader` will give us:
    # (audio_signal, audio_signal_length, transcript_tokens, transcript_length)
    # See the AudioToCharDataset for more details.
    for test_batch in nemo_model_for_speech_recognition.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        targets = test_batch[2]
        targets_lengths = test_batch[3]        
        log_probs, encoded_len, greedy_predictions = nemo_model_for_speech_recognition(
            input_signal=test_batch[0], input_signal_length=test_batch[1]
        )

        if wbs is not None:
            predictions = wbs.compute(log_probs)

        #print (greedy_predictions)
        #print (targets)
        #print (targets_lengths)
        #print (targets.long().cpu().shape)
        # Notice the model has a helper object to compute WER
        nemo_model_for_speech_recognition._wer(greedy_predictions, targets, targets_lengths)
        _, wer_num, wer_denom = nemo_model_for_speech_recognition._wer.compute()
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())

    # We need to sum all numerators and denominators first. Then divide.
    print(f"WER = {sum(wer_nums)/sum(wer_denoms)}")



# Performs a transcription on any file
#
def perform_transcription_on_file(audio_file):
    global nemo_model_for_speech_recognition, nemo_model_params
    
    y, sr = librosa.load(audio_file)
    sf.write("data/output.wav", y, samplerate=sr)
    files = ["data/output.wav"]
    nemo_model_for_speech_recognition.eval()
    for fname, transcription in zip(files, nemo_model_for_speech_recognition.transcribe(paths2audio_files=files)):
        print(f"Audio transcription as: {transcription}")

    
def get_model():
    return nemo_model_for_speech_recognition
  
def get_params():
    return nemo_model_params

# Returns an audio recorder to be displayed.
#
def display_audio_recorder_local():
    global audio_recorder
    camera = CameraStream(constraints=
                          {'facing_mode': 'user',
                           'audio': True,
                           'video': False
                           })
    audio_recorder = AudioRecorder(stream=camera, codecs="pcm")
    return audio_recorder

# Saves the recorded audio into a file
#
def save_recorded_audio_local(file = 'data/output.wav'):
    audio_recorder.save('data/output.webm')

    os.system("ffmpeg -i data/output.webm -hide_banner -loglevel panic -y " + file)
    return file


# Display the audio recorder for colab
#
def display_audio_recorder_colab():
    global colab_sample_rate, colab_audio
  
    AUDIO_HTML = """
    <script>
    var my_div = document.createElement("DIV");
    var my_p = document.createElement("P");
    var my_btn = document.createElement("BUTTON");
    var t = document.createTextNode("Press to start recording");

    my_btn.appendChild(t);
    my_div.appendChild(my_btn);
    document.body.appendChild(my_div);

    var base64data = 0;
    var reader;
    var recorder, gumStream;
    var recordButton = my_btn;
    var state = 0;

    recordButton.innerText = "Start Recording";
    recordButton.disabled = true;

    navigator.mediaDevices.getUserMedia({audio: true}).then((stream) => {
      gumStream = stream;
      var options = {
        mimeType : 'audio/webm;codecs=opus'
      };              
      recorder = new MediaRecorder(stream, options);
      recordButton.disabled = false;
    });

    // https://stackoverflow.com/a/951057
    function sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    var data = new Promise(resolve =>
      {
        console.log('when')

        recordButton.onclick = () => {
          if (state == 0)
          {
            recorder.ondataavailable = function(e) {            
              var url = URL.createObjectURL(e.data);
              var preview = document.createElement('audio');
              preview.controls = true;
              preview.src = url;
              document.body.appendChild(preview);

              reader = new FileReader();
              reader.readAsDataURL(e.data); 
              reader.onloadend = function() {
                base64data = reader.result;

                // Return the result to the eval_js function
                //
                resolve(base64data.toString())

                recordButton.innerText = "Complete";
                recordButton.disabled = true;
                state = 2;            
              }
            };
            recorder.start();
            recordButton.innerText = "Stop Recording";
            state = 1;
          }
          else if (state == 1)
          {
            if (recorder && recorder.state == "recording") {
                recorder.stop();
                gumStream.getAudioTracks()[0].stop();
                recordButton.innerText = "Saving the recording... pls wait!"
            }
          }
        }

      });

    </script>
    """

    display(HTML(AUDIO_HTML))
    data = eval_js("data")
    binary = b64decode(data.split(',')[1])

    process = (ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
    )
    output, err = process.communicate(input=binary)

    riff_chunk_size = len(output) - 8
    # Break up the chunk size into four bytes, held in b.
    q = riff_chunk_size
    b = []
    for i in range(4):
        q, r = divmod(q, 256)
        b.append(r)

    # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
    riff = output[:4] + bytes(b) + output[8:]

    colab_sample_rate, colab_audio = wav_read(io.BytesIO(riff))

# Save the audio for colab
#
def save_recorded_audio_colab(file = 'data/output.wav'):
    sf.write(file, colab_audio, colab_sample_rate)
    return file


# Display the audio recorder whether it is running
# on your local machine or on Google Colab.
# 
def display_audio_recorder():
    if (not IN_COLAB):
        display_audio_recorder_local()
    else:
        display_audio_recorder_colab()

# Save the audio recorded through IPyWebRTC
# on your local machine or on Google Colab.
#
def save_recorded_audio(file = 'data/output.wav'):
    if (not IN_COLAB):
        return save_recorded_audio_local(file)
    else:
        return save_recorded_audio_colab(file)


# Extracts an audio clip from a movie file
#
def extract_audio_clip(movie_file, start_time, end_time, output_file='data/output.wav'):

    # Load the video file.
    #
    videoclip = VideoFileClip(movie_file)

    # Extracts the audio from <start> to <end> seconds
    #
    audioclip = videoclip.audio.subclip(start_time, end_time)

    # Save the audio clip into the audio_final.wav file
    #
    audioclip.write_audiofile(output_file, fps=16000)
    
    return output_file
