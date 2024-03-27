import streamlit as st
from paddleocr import PaddleOCR
from langchain_community.llms import Ollama
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np
import io
import speech_recognition as sr
from requests import post
import wave
import warnings 

  
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

# Initialize OCR model and language model
ocr_model = PaddleOCR(lang='en', use_angle_cls=True)
llm_model = Ollama(model="llama2-uncensored")

# Function to send data to CRM API
@st.cache_data
def send_to_crm(data):
    # Replace these values with actual CRM API endpoint and credentials
    api_endpoint = "https://example.com/api/contacts"
    api_key = "your_api_key"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = post(api_endpoint, json=data, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()  # Return JSON response from CRM API
    except Exception as e:
        st.error(f"Error sending data to CRM: {e}")

# Function to perform OCR on an image
@st.cache_data
def ocr_with_paddle(img):
    final_text = ''
    result = ocr_model.ocr(img)
    for i in range(len(result[0])):
        text = result[0][i][1][0]
        final_text += ' '+ text
    return final_text

# Function to convert speech to text

@st.cache_data        
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Unable to transcribe the audio"
        except sr.RequestError as e:
            return f"Error: {e}"

# Function to process text using language model
# @st.cache_resource
# def process_text(text_input):
#     string=f"""read this text..(.{text_input}..) present inside small bracket....carefully and only give names,email,phone number,address and other important information in json format present in the given text present inside small bracket. you have to only return name, email, address, phone number from the text in json format.... \
              
#             """
#     print(string)
#     processed_result = llm_model.invoke(string)
#     print(processed_result)
#     return processed_result

# x=process_text("mrcro p NAME SURNAME Job Position COMPANY NAME +01234567899 SLOGAN +01234567899 GOES HERE websitename.com websitename.com lorem@ipsum.com lorem@ipsum.com")



def record_sound(duration=5, samplerate=44100):
    st.write("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording complete.")
    return recording

def main():
    st.title("Image Text Extraction and Processing")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Convert uploaded file to bytes
        file_bytes = uploaded_file.read()

        # Perform OCR on the uploaded image
        ocr_result = ocr_with_paddle(file_bytes)

        # Display OCR result
        st.subheader("OCR Result:")
        st.text(ocr_result)

        # print(f"""read the text....{ocr_result}...carefully and give names,email,phone number,address and other important information in json format present the text. you have to only return name, email, address, phone number.... \
        #         for example:
        #         text is like this... BITVATO MICHAL JOHNS Solution Manager Street Address Here Singapore, 2222 urname@email.com BITVATO +18 2767 9470 1808 +18 2767 9470 1808.... \
        #         your output should be simply like this...in json Name: MICHAL JOHNS,
        #                                                     Designation: Solution Manager,
        #                                                     Email: urname@email.com
        #                                                     Address: Singapore, 2222,
        #                                                     Company Name: BITVATO,
        #                                                     Phone No: +18 2767 9470 1808 +18 2767 9470 1808
        #     """)

        # Process OCR result using language model
        st.subheader("Processed Text:")
        processed_result = llm_model.invoke(f"""read this text..(.{ocr_result}..) present inside small bracket....and return only email,phone number,address,name in json format
            the text is this ({ocr_result})""")
        st.text(processed_result)
        st.write(processed_result)

        

        st.title("Real-time Speech to Text")

        duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

        if st.button("Start Recording"):
            audio_data = record_sound(duration=duration, samplerate=44100)
            st.write("Recording complete.")

            # Save audio data to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                sf.write(temp_audio_file.name, audio_data, 44100, 'PCM_24')

            st.subheader("Transcribed Text:")
            text = speech_to_text(temp_audio_file.name)
            st.write(text)

            
            st.subheader("Processed Text Final:")

            st.write(llm_model.invoke(f"""this is the final result or output in json format.... {processed_result}.... and this is the required changes... {text}...you have to do in {processed_result}...do these changes on final result text with required changes and return the final output with changes."""))

if __name__ == "__main__":
   main()




  # for example:
                # text is like this... BITVATO MICHAL JOHNS Solution Manager Street Address Here Singapore, 2222 urname@email.com BITVATO +18 2767 9470 1808 +18 2767 9470 1808.... \
                # your output should be simply like this...in json Name: MICHAL JOHNS,
                #                                             Designation: Solution Manager,
                #                                             Email: urname@email.com
                #                                             Address: Singapore, 2222,
                #                                             Company Name: BITVATO,
                #                                             Phone No: +18 2767 9470 1808 +18 2767 9470 1808
   

   # duration = st.slider("Recording Duration (seconds)", 1, 30, 10)

            # transcribed_text = ""  # Initialize transcribed_text variable

            # if st.button("Start Recording"):
            #     st.write("Recording...")
            #     audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
            #     sd.wait()
            #     st.write("Recording complete.")

            #     # Save audio data to a temporary WAV file
            #     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            #         wf = wave.open(temp_audio_file.name, 'wb')
            #         wf.setnchannels(1)
            #         wf.setsampwidth(2)
            #         wf.setframerate(44100)
            #         wf.writeframes(audio_data.tobytes())
            #         wf.close()

            #         st.subheader("Transcribed Text:")
            #         recognizer = sr.Recognizer()
            #         with sr.AudioFile(temp_audio_file.name) as source:
            #             audio = recognizer.record(source)
            #             try:
            #                 transcribed_text = recognizer.recognize_google(audio)
            #                 st.write(transcribed_text)
            #             except sr.UnknownValueError:
            #                 st.write("Unable to transcribe the audio")
            #             except sr.RequestError as e:
            #                 st.write(f"Error: {e}")