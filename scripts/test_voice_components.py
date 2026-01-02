import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

poc_dir = r"C:\Users\Admin\Documents\voice-ai-poc"

# ========== TEST 1: Deepgram STT ==========
async def test_deepgram():
    print("=" * 60)
    print("TEST 1: Deepgram STT")
    print("=" * 60)
    
    from deepgram import DeepgramClient
    
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("❌ DEEPGRAM_API_KEY not found")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        client = DeepgramClient(api_key)
        print("✅ Deepgram client created")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ========== TEST 2: Cartesia TTS ==========
async def test_cartesia():
    print("\n" + "=" * 60)
    print("TEST 2: Cartesia TTS")
    print("=" * 60)
    
    from cartesia import Cartesia
    
    api_key = os.getenv("CARTESIA_API_KEY")
    voice_id = os.getenv("CARTESIA_VOICE_ID")
    
    if not api_key:
        print("❌ CARTESIA_API_KEY not found")
        return False
    
    print(f"✅ API Key found: {api_key[:15]}...")
    print(f"✅ Voice ID: {voice_id}")
    
    try:
        client = Cartesia(api_key=api_key)
        print("✅ Cartesia client created")
        
        print("\n🔊 Generating test speech...")
        
        # Generate audio (returns generator)
        audio_generator = client.tts.bytes(
            model_id="sonic-english",
            transcript="Hello! I am your healthcare assistant. How can I help you today?",
            voice={
                "mode": "id",
                "id": voice_id
            },
            output_format={
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 24000
            }
        )
        
        # Collect all bytes from generator
        audio_chunks = []
        for chunk in audio_generator:
            audio_chunks.append(chunk)
        audio_data = b''.join(audio_chunks)
        
        # Save audio
        output_path = os.path.join(poc_dir, "test_audio.wav")
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        print(f"✅ Audio saved: {output_path}")
        print(f"✅ Audio size: {len(audio_data)} bytes")
        print("🎧 Play test_audio.wav to hear the voice!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== TEST 3: Twilio ==========
def test_twilio():
    print("\n" + "=" * 60)
    print("TEST 3: Twilio")
    print("=" * 60)
    
    from twilio.rest import Client
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    phone_number = os.getenv("TWILIO_PHONE_NUMBER")
    
    if not account_sid or not auth_token:
        print("❌ Twilio credentials not found")
        return False
    
    print(f"✅ Account SID: {account_sid[:10]}...")
    print(f"✅ Phone Number: {phone_number}")
    
    try:
        client = Client(account_sid, auth_token)
        account = client.api.accounts(account_sid).fetch()
        print(f"✅ Account Status: {account.status}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ========== TEST 4: Azure OpenAI ==========
def test_azure_openai():
    print("\n" + "=" * 60)
    print("TEST 4: Azure OpenAI")
    print("=" * 60)
    
    from openai import AzureOpenAI
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    if not api_key:
        print("❌ AZURE_OPENAI_API_KEY not found")
        return False
    
    print(f"✅ Endpoint: {endpoint}")
    print(f"✅ Deployment: {deployment}")
    
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=endpoint
        )
        
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say hello in 5 words"}],
            max_tokens=20
        )
        
        print(f"✅ Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ========== RUN ALL TESTS ==========
async def main():
    print("\n🧪 VOICE COMPONENTS TEST\n")
    
    results = {
        'Deepgram': await test_deepgram(),
        'Cartesia': await test_cartesia(),
        'Twilio': test_twilio(),
        'Azure OpenAI': test_azure_openai()
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for component, status in results.items():
        print(f"  {'✅' if status else '❌'} {component}")
    
    if all(results.values()):
        print("\n🎉 All components ready!")
    else:
        print("\n⚠️ Some components failed.")

if __name__ == "__main__":
    asyncio.run(main())