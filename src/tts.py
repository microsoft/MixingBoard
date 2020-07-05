import requests, time, re, pyaudio, wave, os
from xml.etree import ElementTree
from shared import get_api_key


class TextToSpeech:

    def __init__(self):
        region, key = get_api_key('speech')
        self.token_url = "https://%s.api.cognitive.microsoft.com/sts/v1.0/issueToken"%region
        self.token_headers = {'Ocp-Apim-Subscription-Key': key}
        self.tts_url = 'https://%s.tts.speech.microsoft.com/cognitiveservices/v1'%region
        self.tts_headers = {
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'riff-24khz-16bit-mono-pcm',
            }
        response = requests.post(self.token_url, headers=self.token_headers)
        self.tts_headers['Authorization'] = 'Bearer ' + str(response.text)

        self.fld_out = 'voice'
        os.makedirs(self.fld_out, exist_ok=True)


    def get_audio(self, txt, name='en-US-JessaNeural'):
        # see: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support

        txt_fname = re.sub(r"[^A-Za-z0-9]", "", txt).lower()
        txt_fname = txt_fname[:min(20, len(txt_fname))]
        path_out = self.fld_out + '/%s_%s.wav'%(txt_fname, name)

        lang = '-'.join(name.split('-')[:2])
        xml_body = ElementTree.Element('speak', version='1.0')
        xml_body.set('{http://www.w3.org/XML/1998/namespace}lang', lang)
        voice = ElementTree.SubElement(xml_body, 'voice')
        voice.set('{http://www.w3.org/XML/1998/namespace}lang', lang)
        voice.set('name', name)
        voice.text = txt
        body = ElementTree.tostring(xml_body)

        response = requests.post(self.tts_url, headers=self.tts_headers, data=body)
        if response.status_code == 200:
            with open(path_out, 'wb') as audio:
                audio.write(response.content)
            return path_out
        else:
            print('[TTS] failed with status code: ' + str(response.status_code))
            return None


    def open_audio(self, path_audio):
        if path_audio is None:
            return
              
        chunk = 1024  
        f = wave.open(path_audio,"rb")  
        p = pyaudio.PyAudio()  
        stream = p.open(format=p.get_format_from_width(f.getsampwidth()),  
                        channels=f.getnchannels(),  
                        rate=f.getframerate(),  
                        output=True)  
        data = f.readframes(chunk)  
        while data:  
            stream.write(data)  
            data = f.readframes(chunk)  
        stream.stop_stream()  
        stream.close()  
        p.terminate()


def play_tts():
    tts = TextToSpeech()
    while True:
        txt = input('\nTXT:\t')
        if len(txt) == 0:
            break
        path_audio = tts.get_audio(txt)
        if path_audio is not None:
            print('audio saved to '+path_audio)
        tts.open_audio(path_audio)


if __name__ == "__main__":
    play_tts()