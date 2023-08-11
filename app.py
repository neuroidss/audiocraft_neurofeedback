# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import warnings

import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen

import julius

import torchaudio
import math

MODEL = None  # Last used model
IS_BATCHED = "facebook/MusicGen" in os.environ.get('SPACE_ID', '')
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call

time1start = None
time2start = None
time3start = None

def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomitting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True

def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out

def load_model(version='melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = MusicGen.get_pretrained(version)


def update_coherence_json_click(attention_type, coherence_json):
    if not (MODEL is None):
      MODEL.lm.transformer._set_attention_coherence(attention_type, coherence_json)
      MODEL.lm.transformer.attention_type = attention_type
      MODEL.lm.transformer.coherence_json = coherence_json
      MODEL.attention_type = attention_type
      MODEL.coherence_json = coherence_json
#      print("coherence_json: ", coherence_json)


def _do_predictions(texts, melodies, duration, progress=False, **gen_kwargs):
    MODEL.set_generation_params(duration=duration/MODEL.divider, extend_stride = MODEL.extend_stride, **gen_kwargs)
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            
            
            if MODEL.pitch_shift:
              melody = torchaudio.functional.pitch_shift(waveform=melody, sample_rate=sr, n_steps=int(math.log2(MODEL.divider)*24), bins_per_octave=24, n_fft=512, win_length=None, hop_length=None, window=None)

            melody = convert_audio(melody, sr*MODEL.divider, target_sr, target_ac)
            processed_melodies.append(melody)

#    print('MODEL.generate')

    if any(m is not None for m in processed_melodies):
        outputs = MODEL.generate_with_chroma(
            descriptions=texts,
            melody_wavs=processed_melodies,
            melody_sample_rate=target_sr,
            progress=progress,
        )
    else:
        outputs = MODEL.generate(texts, progress=progress)

            
    outputs = outputs.detach().cpu().float()
    out_files = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        
#            print(output)
#            output = julius.resample_frac(output, int(MODEL.sample_rate), int(MODEL.sample_rate*dividier))
        

            if MODEL.pitch_shift:
              print(output)
              output = torchaudio.functional.pitch_shift(waveform=output, sample_rate=int(MODEL.sample_rate/MODEL.divider), n_steps=int(math.log2(MODEL.divider)*24), bins_per_octave=24, n_fft=512, win_length=None, hop_length=None, window=None)

#            output = output.detach().cpu().float()

            audio_write(
                file.name, output, int(MODEL.sample_rate/MODEL.divider), strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            out_files.append(file.name)
#            out_files.append(pool.submit(make_waveform, file.name))
#    res = [out_file.result() for out_file in out_files]
    res = [out_file for out_file in out_files]
    print("batch finished", len(texts), time.time() - be)
    return res


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('melody')
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return [res]


def predict_full(model, text, melody, duration, divider, pitch_shift, sampler, time_shift, max_duration, extend_stride, attention_type, coherence_json, seed, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    load_model(model)

    MODEL.make_waveform = make_waveform
    MODEL.divider = divider
    MODEL.pitch_shift = pitch_shift
    MODEL.pool = pool
    MODEL.sampler = sampler
    MODEL.time_shift = time_shift
    MODEL.max_duration = max_duration
    MODEL.extend_stride = extend_stride
    MODEL.attention_type = attention_type
    MODEL.coherence_json = coherence_json
#    MODEL.lm.transformer
    MODEL.tmp = []
    MODEL.tmp_new = False
    MODEL.audio_data = []
    MODEL.audio_data_new = False
    MODEL.audio_tmp = []
    MODEL.audio_tmp_new = False
    global time1start
    MODEL.time1start = time1start
    
    if seed < 0:
        import random
        seed = random.randint(0, 0xffff_ffff_ffff)
    torch.manual_seed(seed)

    def _progress(generated, to_generate):
        progress((generated, to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    outs = _do_predictions(
        [text], [melody], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
    print(outs[0])
    return [outs[0], seed]

def audio_stream(mic):
    if not (mic is None):
     print('mic: '+str(mic))
#     yield mic
     return mic
     
     if not (MODEL is None):
      if MODEL.audio_data_new:
          MODEL.audio_data_new = False
#          audio_data = b""
#          audio_data += MODEL.audio_data[len(MODEL.audio_data)-1]
          print('audio_data: '+str(audio_data))
          audio_data = MODEL.audio_data[len(MODEL.audio_data)-1]
#          yield mic
#          return audio_data
def check_tmp1(sampler):
    import time
    global time1start
    if time1start is None:
      time1start = time.time()
    if not (MODEL is None):
     MODEL.time1start = time1start
     if hasattr(MODEL, "tmp"):
      if (len(MODEL.tmp)>0) and (MODEL.sampler>=1):
       if hasattr(MODEL, "tmp_new") and hasattr(MODEL, "frame_idx"):
#        while not(MODEL.tmp_new and ((len(MODEL.tmp))%MODEL.sampler==0)):
        while not(MODEL.tmp_new and (((MODEL.frame_idx))%MODEL.sampler==0)):
          time.sleep(0.01)
        if MODEL.tmp[len(MODEL.tmp)-1].done():
          tmp_result = MODEL.tmp[len(MODEL.tmp)-1].result()
          MODEL.tmp_new = False
          print('tmp_result: '+str(tmp_result))
          return tmp_result
def check_tmp2(sampler):
    import time
    global time2start
    if time2start is None:
      time2start = time.time()
    if not (MODEL is None):
     MODEL.time2start = time2start
     if hasattr(MODEL, "tmp"):
      if (len(MODEL.tmp)>0) and (MODEL.sampler>=2):
       if hasattr(MODEL, "tmp_new") and hasattr(MODEL, "frame_idx"):
#        while not(MODEL.tmp_new and ((len(MODEL.tmp))%MODEL.sampler==1)):
        while not(MODEL.tmp_new and (((MODEL.frame_idx))%MODEL.sampler==1)):
          time.sleep(0.01)
        if MODEL.tmp[len(MODEL.tmp)-1].done():
          tmp_result = MODEL.tmp[len(MODEL.tmp)-1].result()
          MODEL.tmp_new = False
          print('tmp_result: '+str(tmp_result))
          return tmp_result
def check_tmp3(sampler):
    import time
    global time3start
    if time3start is None:
      time3start = time.time()
    if not (MODEL is None):
     MODEL.time3start = time3start
     if hasattr(MODEL, "tmp"):
      if (len(MODEL.tmp)>0) and (MODEL.sampler>=3):
       if hasattr(MODEL, "tmp_new") and hasattr(MODEL, "frame_idx"):
#        while not(MODEL.tmp_new and ((len(MODEL.tmp))%MODEL.sampler==2)):
        while not(MODEL.tmp_new and (((MODEL.frame_idx))%MODEL.sampler==2)):
          time.sleep(0.01)
        if MODEL.tmp[len(MODEL.tmp)-1].done():
          tmp_result = MODEL.tmp[len(MODEL.tmp)-1].result()
          MODEL.tmp_new = False
          print('tmp_result: '+str(tmp_result))
          return tmp_result
def check_audio_tmp1(sampler):
    import time
    global time1start
    if time1start is None:
      time1start = time.time()
    if not (MODEL is None):
     MODEL.time1start = time1start
     if hasattr(MODEL, "tmp"):
      if (len(MODEL.audio_tmp)>0) and (MODEL.sampler>=1):
       if hasattr(MODEL, "audio_tmp_new") and hasattr(MODEL, "frame_idx"):
#        while not(MODEL.tmp_new and ((len(MODEL.tmp))%MODEL.sampler==0)):
        while not(MODEL.audio_tmp_new and (((MODEL.frame_idx))%MODEL.sampler==0)):
          time.sleep(0.01)
        audio_tmp_result = MODEL.audio_tmp[len(MODEL.audio_tmp)-1]
        MODEL.audio_tmp_new = False
        print('audio_tmp_result: '+str(audio_tmp_result))
        return audio_tmp_result
def check_audio_tmp2(sampler):
    import time
    global time2start
    if time2start is None:
      time2start = time.time()
    if not (MODEL is None):
     MODEL.time2start = time2start
     if hasattr(MODEL, "tmp"):
      if (len(MODEL.audio_tmp)>0) and (MODEL.sampler>=2):
       if hasattr(MODEL, "audio_tmp_new") and hasattr(MODEL, "frame_idx"):
#        while not(MODEL.tmp_new and ((len(MODEL.tmp))%MODEL.sampler==0)):
        while not(MODEL.audio_tmp_new and (((MODEL.frame_idx))%MODEL.sampler==1)):
          time.sleep(0.01)
        audio_tmp_result = MODEL.audio_tmp[len(MODEL.audio_tmp)-1]
        MODEL.audio_tmp_new = False
        print('audio_tmp_result: '+str(audio_tmp_result))
        return audio_tmp_result
def check_audio_tmp3(sampler):
    import time
    global time3start
    if time3start is None:
      time3start = time.time()
    if not (MODEL is None):
     MODEL.time3start = time3start
     if hasattr(MODEL, "tmp"):
      if (len(MODEL.audio_tmp)>0) and (MODEL.sampler>=3):
       if hasattr(MODEL, "audio_tmp_new") and hasattr(MODEL, "frame_idx"):
#        while not(MODEL.tmp_new and ((len(MODEL.tmp))%MODEL.sampler==0)):
        while not(MODEL.audio_tmp_new and (((MODEL.frame_idx))%MODEL.sampler==2)):
          time.sleep(0.01)
        audio_tmp_result = MODEL.audio_tmp[len(MODEL.audio_tmp)-1]
        MODEL.audio_tmp_new = False
        print('audio_tmp_result: '+str(audio_tmp_result))
        return audio_tmp_result

def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MusicGen
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", value="psytrance", interactive=True)
                    melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="small", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=120000, value=300, label="Duration", interactive=True)
                with gr.Row():
                    divider = gr.Slider(minimum=0.1, maximum=120, value=1.8, step=0.1, label="Divider", interactive=True, elem_id="divider")
                    pitch_shift = gr.Checkbox(label="pitch_shift", info="pitch_shift")
                with gr.Row():
                    sampler = gr.Slider(minimum=0, maximum=3, value=3, step=1, label="Sampler", interactive=False)
                    time_shift = gr.Slider(minimum=0, maximum=10, value=2.7, step=0.1, label="Time Shift", interactive=True)
                with gr.Row():
                    max_duration = gr.Slider(minimum=1, maximum=300, value=1.62, step=0.01, label="max_duration", interactive=True, elem_id="max_duration")
                with gr.Row():
                    extend_stride = gr.Slider(minimum=0.1, maximum=180, value=1, step=0.01, label="extend_stride (<max_duration)", interactive=True)
                with gr.Row():
                    attention_type = gr.Radio(["default", "random", "coherence"], label="Attention Type", value="default", interactive=True)
                    coherence_json = gr.Text(label="Coherence JSON", value="{}", interactive=True)
                    update_coherence_json = gr.Button("Update Coherence JSON")
                with gr.Row():
                    seed = gr.Number(label="Seed", value=-1, scale=4, precision=0, interactive=True)
                    gr.Button('\U0001f3b2\ufe0f', scale=1).click(fn=lambda: -1, outputs=[seed], queue=False)
                    reuse_seed = gr.Button('\u267b\ufe0f', scale=1)
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, step=0.0001, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
#                output = gr.Video(label="Generated Music")
#                output = gr.Video(label="Generated Music (updates every second)")#, autoplay=True)
#                output1 = gr.Video(label="Generated Music (updates every second)")
#                output2 = gr.Video(label="Generated Music (updates every second)")
#                output3 = gr.Video(label="Generated Music (updates every second)")
#                output1 = gr.Video(label="Generated Music (updates every second)", autoplay=True)
#                output2 = gr.Video(label="Generated Music (updates every second)", autoplay=True)
#                output3 = gr.Video(label="Generated Music (updates every second)", autoplay=True)
                output1 = gr.Audio(label="Generated Music (updates)")
                output2 = gr.Audio(label="Generated Music (updates)")
                output3 = gr.Audio(label="Generated Music (updates)")
                output = gr.Audio(label="Generated Music")
#                output_audio = gr.Audio(label="Generated Music (updates every second)", autoplay=True)
#                output1 = gr.Audio(label="Generated Music (streaming)")
#                input1 = gr.Audio(source="microphone", type="numpy", streaming=True)
                seed_used = gr.Number(label='Seed used', value=-1, interactive=False)

                
        submit.click(predict_full, inputs=[model, text, melody, duration, divider, pitch_shift, sampler, time_shift, max_duration, extend_stride, attention_type, coherence_json, seed, topk, topp, temperature, cfg_coef], outputs=[output, seed_used])

        update_coherence_json.click(update_coherence_json_click, inputs=[attention_type, coherence_json], queue=False)

        reuse_seed.click(fn=lambda x: x, inputs=[seed_used], outputs=[seed], queue=False)
        
#        output_audio = interface.load(check_audio_tmp, None, outputs=[output_audio], every=1)
#        input1.stream(audio_stream, inputs=[input1], outputs=[output1])
#        output1 = interface.load(audio_stream, inputs=[input1], outputs=[output1], live=True)
#        output1 = interface.load(check_tmp1, None, outputs=[output1], every=1.5)
#        output1 = interface.load(check_tmp1, inputs=[sampler], outputs=[output1], every=0.05, _js="(x) => console.log(Date.now())")

        import time
        global time1start
        if time1start is None:
          time1start = time.time()

        output1 = interface.load(check_audio_tmp1, inputs=[sampler], outputs=[output1], every=0.05, _js="(sampler) => {var time1start = "+str(time1start*1000)+"; function playAudio1(){ var audioList = document.getElementsByTagName('audio'); var progress_levelList = document.getElementsByClassName('progress-level'); if((audioList.length>=sampler)&&(progress_levelList.length>0)) audioList[0].play(); var time1now = Date.now(); var frame_time1 = parseInt(document.getElementById('divider').querySelector('input').value*sampler*1000); var next_time1 = frame_time1-((time1now-time1start)%frame_time1); setTimeout(() => { playAudio1() }, next_time1) }; setTimeout(() => { playAudio1() }, 1000*0*document.getElementById('divider').querySelector('input').value) }")
        output2 = interface.load(check_audio_tmp2, inputs=[sampler], outputs=[output2], every=0.05, _js="(sampler) => {var time2start = "+str(time1start*1000)+"; function playAudio2(){ var audioList = document.getElementsByTagName('audio'); var progress_levelList = document.getElementsByClassName('progress-level'); if((audioList.length>=sampler)&&(progress_levelList.length>0)) audioList[1].play(); var time2now = Date.now(); var frame_time2 = parseInt(document.getElementById('divider').querySelector('input').value*sampler*1000); var next_time2 = frame_time2-((time2now-(time2start+1000*1*document.getElementById('divider').querySelector('input').value))%frame_time2); setTimeout(() => { playAudio2() }, next_time2) }; setTimeout(() => { playAudio2() }, 1000*1*document.getElementById('divider').querySelector('input').value) }")
        output3 = interface.load(check_audio_tmp3, inputs=[sampler], outputs=[output3], every=0.05, _js="(sampler) => {var time3start = "+str(time1start*1000)+"; function playAudio3(){ var audioList = document.getElementsByTagName('audio'); var progress_levelList = document.getElementsByClassName('progress-level'); if((audioList.length>=sampler)&&(progress_levelList.length>0)) audioList[2].play(); var time3now = Date.now(); var frame_time3 = parseInt(document.getElementById('divider').querySelector('input').value*sampler*1000); var next_time3 = frame_time3-((time3now-(time3start+1000*2*document.getElementById('divider').querySelector('input').value))%frame_time3); setTimeout(() => { playAudio3() }, next_time3) }; setTimeout(() => { playAudio3() }, 1000*2*document.getElementById('divider').querySelector('input').value) }")
#        output1 = interface.load(check_tmp1, inputs=[sampler], outputs=[output1], every=0.05, _js="() => {var time1start = "+str(time1start*1000)+"; function playAudio1(){ var videoList = document.getElementsByTagName('video'); if(videoList.length==3) videoList[0].play(); var time1now = Date.now(); var frame_time1 = parseInt(document.getElementById('divider').querySelector('input').value*3000); var next_time1 = frame_time1-((time1now-time1start)%frame_time1); console.log(Date.now()); setTimeout(() => { playAudio1() }, next_time1) }; setTimeout(() => { playAudio1() }, 1000*0*document.getElementById('divider').querySelector('input').value) }")
#        output2 = interface.load(check_tmp2, inputs=[sampler], outputs=[output2], every=0.05, _js="() => {var time2start = "+str(time1start*1000)+"+1000*1*document.getElementById('divider').querySelector('input').value; function playAudio2(){ var videoList = document.getElementsByTagName('video'); if(videoList.length==3) videoList[1].play(); var time2now = Date.now(); var frame_time2 = parseInt(document.getElementById('divider').querySelector('input').value*3000); var next_time2 = frame_time2-((time2now-time2start)%frame_time2); console.log(Date.now()); setTimeout(() => { playAudio2() }, next_time2) }; setTimeout(() => { playAudio2() }, 1000*1*document.getElementById('divider').querySelector('input').value) }")
#        output3 = interface.load(check_tmp3, inputs=[sampler], outputs=[output3], every=0.05, _js="() => {var time3start = "+str(time1start*1000)+"+1000*2*document.getElementById('divider').querySelector('input').value; function playAudio3(){ var videoList = document.getElementsByTagName('video'); if(videoList.length==3) videoList[2].play(); var time3now = Date.now(); var frame_time3 = parseInt(document.getElementById('divider').querySelector('input').value*3000); var next_time3 = frame_time3-((time3now-time3start)%frame_time3); console.log(Date.now()); setTimeout(() => { playAudio3() }, next_time3) }; setTimeout(() => { playAudio3() }, 1000*2*document.getElementById('divider').querySelector('input').value) }")

#        output1 = interface.load(check_tmp1, inputs=[sampler], outputs=[output1], every=0.05, _js="() => setTimeout(() => { document.getElementsByTagName('video')[0].play() }, 3000)")
#        output2 = interface.load(check_tmp2, inputs=[sampler], outputs=[output2], every=0.05, _js="() => setTimeout(() => { document.getElementsByTagName('video')[0].play() }, 6000)")
#        output3 = interface.load(check_tmp3, inputs=[sampler], outputs=[output3], every=0.05, _js="() => setTimeout(() => { document.getElementsByTagName('video')[0].play() }, 9000)")
#        output1 = interface.load(check_tmp1, inputs=[sampler], outputs=[output1], every=0.05)
#        output2 = interface.load(check_tmp2, inputs=[sampler], outputs=[output2], every=0.05)
#        output3 = interface.load(check_tmp3, inputs=[sampler], outputs=[output3], every=0.05)
#        output1 = interface.load(check_audio_tmp1, None, outputs=[output1], every=0.1)
#        output2 = interface.load(check_audio_tmp2, None, outputs=[output2], every=0.1)
#        output3 = interface.load(check_audio_tmp3, None, outputs=[output3], every=0.1)
#        output2 = interface.load(predict_full, inputs=[model, text, melody, duration, divider, sampler, topk, topp, temperature, cfg_coef], outputs=[output], every=0.01)
        #, _js=autoplay_audio
#        period1.change(predict_full, inputs=[model, text, melody, duration, divider, topk, topp, temperature, cfg_coef], outputs=[plot1], every=1, cancels=[dep1])

        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                    "melody"
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                    "medium"
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                    "medium",
                ],
            ],
            inputs=[text, melody, model],
            outputs=[output]
        )
        gr.Markdown(
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            The model can generate up to 30 seconds of audio in one pass. It is now possible
            to extend the generation by feeding back the end of the previous chunk of audio.
            This can take a long time, and the model might lose consistency. The model might also
            decide at arbitrary positions that the song ends.

            **WARNING:** Choosing long durations will take a long time to generate (2min might take ~10min). An overlap of 12 seconds
            is kept with the previously generated chunk, and 18 "new" seconds are generated each time.

            We present 4 model variations:
            1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
            2. Small -- a 300M transformer decoder conditioned on text only.
            3. Medium -- a 1.5B transformer decoder conditioned on text only.
            4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.)

            When using `melody`, ou can optionaly provide a reference audio from
            which a broad melody will be extracted. The model will then try to follow both the description and melody provided.

            You can also use your own GPU or a Google Colab by following the instructions on our repo.
            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
            for more details.
            """
        )

#        interface.queue(concurrency_count=10).launch(**launch_kwargs)
        interface.queue().launch(**launch_kwargs)


def ui_batched(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # MusicGen

            This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Describe your music", lines=2, interactive=True)
                    melody = gr.Audio(source="upload", type="numpy", label="Condition on a melody (optional)", interactive=True)
                with gr.Row():
                    submit = gr.Button("Generate")
            with gr.Column():
                output = gr.Video(label="Generated Music")
        submit.click(predict_batched, inputs=[text, melody], outputs=[output], batch=True, max_batch_size=MAX_BATCH_SIZE)
        gr.Examples(
            fn=predict_batched,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
                    "./assets/bach.mp3",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                ],
            ],
            inputs=[text, melody],
            outputs=[output]
        )
        gr.Markdown("""
        ### More details

        The model will generate 12 seconds of audio based on the description you provided.
        You can optionaly provide a reference audio from which a broad melody will be extracted.
        The model will then try to follow both the description and melody provided.
        All samples are generated with the `melody` model.

        You can also use your own GPU or a Google Colab by following the instructions on our repo.

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
        for more details.
        """)

        demo.queue(max_size=8 * 4).launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    # Show the interface
    if IS_BATCHED:
        ui_batched(launch_kwargs)
    else:
        ui_full(launch_kwargs)
