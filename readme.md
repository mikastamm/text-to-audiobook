
# Multi-Engine Text-to-Speech (TTS) System  

This project is designed to create multi-speaker audiobooks from short to medium-length stories. It supports **Kokoro** and **Zonos** voice synthesis and includes some demo voices.

## Features  
✅ Supports **Kokoro** and **Zonos** TTS models

  ☑️ Both can be mixed in the same audio file  
  
✅ Local voice synthesis 

✅ Uses a language model to assign speakers to text & improve speakability

✅ Speaker assignment through `<speaker voice="name">` tags  

✅ Batch processing of multiple files

✅ Loudness normalization for consistent audio levels


## Example

https://github.com/user-attachments/assets/7c779dd9-98ec-46b1-8e96-47e4641816e7

Dialogue starts at second 23

## Installation  

### **1. Clone the Repository**  
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### **2.Build Docker container**  
```sh
docker build -f Dockerfile.base -t audiobook-tts-base .
docker build -f Dockerfile -t audiobook-tts .
```

### **3. (Optional) Connect it to OpenAI or OpenRouter**  
put your open router api key in the `gpt_secret_key` file (If you want to use the OpenAI API, you need to edit the configuration.yaml)

## Basic Usage
1️⃣ **Prepare input files**: Place the `.txt` files you want spoken in the `1-raw-text/` folder, to have speakers assigned automatically. This will prompt the language model to fitting assign fitting speakers to the text and requires the API key to be set up. Alternatively, you can manually assign speakers (See [Adding New Voices](#adding-new-voices)).

2️⃣ **Run the program**:
   **Linux/macOS**:
     ```source start.sh```
   **Windows**:
     ```start.bat```

3️⃣ The generated `.wav` files will be saved in the `3-output/` folder.

## System Requirements
Docker
CUDA capable GPU
~5GB VRAM if using Zonos (You can only use Kokoro if you have less.)

## How do I get a good result?

In my testing I preferred to use Kokoro models for any sort of narrator. The voices `af_heart` and `af_bella` are by far the best as of now.

Zono's models are great for dialogue and conversations between characters as it has quite the high emotional range, but it's a bit too expressive for my tastes as a narrator. 


---

## Annotating Text for Speaker Assignment

### Automatic Annotation
Automatic annotation requires the API key to be set in gpt_secret_key.txt.
For each file in `1-raw-text` it will automatically prompt the language model with the prompt defined in `prompts/edit-raw-story-prompt.md` To annotate speakers and make minor changes such as writing out dates into words.
The result will be written to the `2-annotated-text` folder.

### Manual Annotation

text files placed into the `2-annotated-text` in the following format.

>`This will be spoken by the narrator that is set in the configuration.yml`
>
>`<speaker voice="af_heart">This text will be spoken by a af_heart</speaker>`
>
>`<pause duration='long'> This will be spoken after a long pause `
>
>`<pause duration='short'> This will be spoken after a short pause` 

## Adding New Voices  

The project already includes some zonos speakers. However, since only short reference audios, generated from major tts provider are included, their quality is not as high as it could be. 

Using real recorded audio from a professional speaker in a good quality of a length from 30 to 60 seconds results in much better outputs.

### Zonos

Sonos supports voice cloning given an example of the speaker you would like to clone. Place 10 to 60 second audio clips of them in the speakers directory. 

I recommend you try them out int the [Zonos Playground](https://playground.zyphra.com/sign-in?redirect_url=https%3A%2F%2Fplayground.zyphra.com%2Faudio) to see whether they produce good results 

If you are having a language model assigned speakers it makes sense to give them a very descriptive name including the gender and where you would like to use them.

You can optionally grade the speakers in the filename which will affect how they are presented in the prompt. 
`s_*.wav`: S-Tier, will be used most often and for main characters
`a_*.wav`: A-Tier, will be used for secondary characters
`b_*.wav`: B-Tier, will be used as a fallback for when you run out.
`x_*.wav`: X-Tier, will be used for special voices (think evil, robot etc.)`

### Kokoro

Kokoro does not support adding new voices, though new voices might be introduced in updates to Kokoro.
