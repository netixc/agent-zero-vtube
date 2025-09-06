"""
VTube Simple Extension - Simplified version for stability

Sends complete Agent-Zero responses to VTube without duplicates.
"""

from python.helpers.extension import Extension
from agent import LoopData
import requests
import base64
import re
import time
import asyncio
from openai import OpenAI
import wave
import io
import os


class VTubeSimple(Extension):
    """Simple VTube integration that just works"""
    
    # Class-level tracking to persist across instances
    _sent_messages = set()
    _last_clear_time = time.time()
    _pending_response = ""
    _last_update_time = 0
    _send_task = None
    
    # Configuration
    VTUBE_API_URL = os.getenv("VTUBE_API_URL", "http://192.168.50.x:12393")
    TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://192.168.50.x:8880/v1")
    TTS_VOICE = os.getenv("TTS_VOICE", "af_sky+af_bella")
    TTS_MODEL = os.getenv("TTS_MODEL", "kokoro")
    
    def __init__(self, agent):
        super().__init__(agent)
        
    async def execute(self, loop_data=None, text="", parsed=None, **kwargs):
        """Send complete responses to VTube"""
        # Skip if no parsed data
        if not parsed or not isinstance(parsed, dict):
            return
            
        # Check if response tool
        if parsed.get("tool_name") != "response":
            return
            
        # Get text from tool args
        tool_args = parsed.get("tool_args")
        if not tool_args or not isinstance(tool_args, dict):
            return
            
        response_text = tool_args.get("text", "").strip()
        if not response_text:
            return
            
        # Update pending response and timestamp
        VTubeSimple._pending_response = response_text
        VTubeSimple._last_update_time = time.time()
        
        # Cancel any existing send task
        if VTubeSimple._send_task and not VTubeSimple._send_task.done():
            VTubeSimple._send_task.cancel()
        
        # Schedule a new send task after delay
        VTubeSimple._send_task = asyncio.create_task(self._delayed_send())
    
    async def _delayed_send(self):
        """Send response after a delay to ensure it's complete"""
        try:
            # Simple delay like the old extension - just wait 0.5s
            await asyncio.sleep(0.5)
            
            # Get the current response text
            response_text = VTubeSimple._pending_response
            if not response_text:
                return
                
            # Check if already sent
            if response_text in VTubeSimple._sent_messages:
                return
                
            # Check if this is a partial of something already sent
            for sent in VTubeSimple._sent_messages:
                if response_text in sent and response_text != sent:
                    return
                    
            # Check if we already sent something that contains this
            for sent in VTubeSimple._sent_messages:
                if sent in response_text and response_text != sent:
                    # Remove the old partial message
                    VTubeSimple._sent_messages.discard(sent)
                    break
            
            # Add to sent messages
            VTubeSimple._sent_messages.add(response_text)
            
            # Clear old messages periodically (every 5 minutes)
            current_time = time.time()
            if current_time - VTubeSimple._last_clear_time > 300:
                VTubeSimple._sent_messages.clear()
                VTubeSimple._last_clear_time = current_time
                VTubeSimple._sent_messages.add(response_text)
            
            # Send to VTube
            await self._send_to_vtube(response_text)
            
        except asyncio.CancelledError:
            # Task was cancelled, that's fine
            pass
        except Exception as e:
            if str(e).strip():
                self.agent.context.log.log(
                    type="error",
                    heading="VTube Error",
                    content=str(e)[:100]
                )
    
    async def _send_to_vtube(self, response_text):
        """Actually send the response to VTube"""
        try:
            # Get emotion for the whole response
            emotion = self._get_emotion(response_text)
            
            # Send the complete response with detected emotion
            await self._send_single_emotion(emotion, response_text)
                
        except Exception as e:
            if str(e).strip():
                self.agent.context.log.log(
                    type="error",
                    heading="VTube Error",
                    content=f"Send error: {str(e)[:100]}"
                )
    
    async def _send_single_emotion(self, emotion, text):
        """Send a single emotion and text to VTube"""
        try:
            text_with_emotion = f"{emotion} {text}"
            
            # Log what we're doing
            self.agent.context.log.log(
                type="info",
                heading="VTube",
                content=f"Sending: {text_with_emotion[:60]}..."
            )
            
            # Generate audio
            audio_data = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_audio, text
            )
            if not audio_data:
                return
                
            # Simple volume pattern like old extension - faster
            duration_ms = len(audio_data) // 88  # Rough estimate
            num_chunks = max(1, duration_ms // 20)
            volumes = [0.7] * num_chunks
            
            # Send to VTube
            payload = {
                "type": "external_audio",
                "audio": base64.b64encode(audio_data).decode('utf-8'),
                "volumes": volumes,
                "slice_length": 20,
                "display_text": {
                    "text": text_with_emotion,
                    "duration": len(volumes) * 0.02
                },
                "source": "agent_zero_extension"
            }
            
            response = requests.post(
                f"{self.VTUBE_API_URL}/api/external_audio",
                json=payload,
                timeout=5
            )
            
            if response.status_code != 200:
                self.agent.context.log.log(
                    type="error",
                    heading="VTube Error",
                    content=f"API returned {response.status_code}"
                )
                
        except Exception as e:
            if str(e).strip():
                self.agent.context.log.log(
                    type="error",
                    heading="VTube Error",
                    content=f"Send error: {str(e)[:100]}"
                )
    
    def _get_emotion(self, text):
        """Enhanced emotion detection supporting all VTube model emotions"""
        text_lower = text.lower()
        
        # Check if emotion tag already exists in text
        emotion_tags = re.findall(r'\[(\w+)\]', text)
        if emotion_tags:
            # Return the first emotion tag found
            return f"[{emotion_tags[0]}]"
        
        # When user asks to show faces/emotions, show joy as default
        if "faces" in text_lower and "make" in text_lower:
            return "[joy]"
        
        # Map emojis to emotions - comprehensive mapping
        emoji_emotion_map = {
            "😀": "[joy]", "😊": "[joy]", "😃": "[joy]", "😄": "[joy]", "😁": "[joy]", "😍": "[joy]", "😂": "[joy]",
            "😢": "[sadness]", "😞": "[sadness]", "😔": "[sadness]", "😭": "[sadness]", 
            "😠": "[anger]", "😡": "[anger]", "😤": "[anger]",
            "😲": "[surprise]", "😮": "[surprise]", "😯": "[surprise]",
            "😨": "[fear]", "😰": "[fear]", "😱": "[fear]",
            "😉": "[smirk]", "😏": "[smirk]", "😎": "[smirk]",
            "🤔": "[smirk]",  # Thinking maps to smirk
            "🤖": "[neutral]",  # Robot maps to neutral
            "😕": "[neutral]", "😐": "[neutral]"
        }
        
        # Check for the FIRST emoji and map to emotion (prioritizes first emoji in response)
        for emoji, emotion in emoji_emotion_map.items():
            if emoji in text:
                return emotion  # Direct mapping - simpler and faster
        
        # Check for emotional content in the response (without explicit face request)
        if any(w in text_lower for w in ["hello", "hi", "great", "wonderful", "awesome", "fantastic"]):
            return "[joy]"
        elif any(w in text_lower for w in ["sorry", "unfortunately", "regret"]):
            return "[sadness]"
        elif any(w in text_lower for w in ["error", "wrong", "failed", "frustrated", "annoying"]):
            return "[anger]"
        elif any(w in text_lower for w in ["wow", "amazing", "incredible", "unbelievable"]):
            return "[surprise]"
        elif any(w in text_lower for w in ["scared", "afraid", "terrified"]):
            return "[fear]"
        elif any(w in text_lower for w in ["hmm", "thinking", "perhaps", "maybe", "consider"]):
            return "[smirk]"
        elif any(w in text_lower for w in ["gross", "disgusting", "ugh", "yuck"]):
            return "[disgust]"
        else:
            return "[neutral]"
    
    def _calculate_volumes(self, audio_data):
        """Calculate realistic volume levels for lip-sync"""
        try:
            # Parse WAV header to get duration
            audio_file = io.BytesIO(audio_data)
            with wave.open(audio_file, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration_ms = (frames / rate) * 1000
                
            # Calculate number of 20ms chunks
            chunk_count = int(duration_ms / 20)
            
            # Create a varying volume pattern for realistic lip-sync
            volumes = []
            for i in range(chunk_count):
                # Create natural speech pattern with variation
                base = 0.5
                variation = 0.3 * abs((i % 10) - 5) / 5  # Wave pattern
                noise = 0.1 * (i % 3) / 3  # Small noise
                volume = min(1.0, base + variation + noise)
                volumes.append(volume)
            
            return volumes if volumes else [0.5]
            
        except Exception as e:
            # Fallback to simple pattern
            duration_estimate = len(audio_data) // 88
            chunk_count = max(1, duration_estimate // 20)
            return [0.5 + 0.2 * (i % 2) for i in range(chunk_count)]
    
    def _generate_audio(self, text):
        """Generate audio with Kokoro TTS"""
        try:
            # Clean text for TTS - remove emojis, markdown, and emotion tags
            clean_text = self._clean_text_for_tts(text)
            
            client = OpenAI(
                api_key="not-needed",
                base_url=self.TTS_BASE_URL
            )
            
            response = client.audio.speech.create(
                model=self.TTS_MODEL,
                voice=self.TTS_VOICE,
                response_format="wav",
                input=clean_text
            )
            
            return response.content
            
        except Exception as e:
            self.agent.context.log.log(
                type="error",
                heading="TTS Error",
                content=f"Failed to generate audio: {str(e)[:100]}"
            )
            return None
    
    def _clean_text_for_tts(self, text):
        """Remove emotion tags, emojis, markdown, and clean text for TTS"""
        import re
        
        # IMPORTANT: Remove emotion tags first (e.g., [joy], [sadness], etc.)
        clean = re.sub(r'\[\w+\]', '', text)
        
        # Remove common emojis and their descriptions
        emoji_patterns = [
            r'😊|😃|😄|😁|🙂|🙃|😉',  # Happy emojis
            r'😢|😭|😞|😔|☹️|🙁',      # Sad emojis
            r'😠|😡|😤|🤬',            # Angry emojis
            r'😲|😮|😯|😦|😧|🤯',      # Surprised emojis
            r'😨|😰|😱',              # Fear emojis
            r'[\U0001F600-\U0001F64F]',  # Emoticons
            r'[\U0001F300-\U0001F5FF]',  # Symbols & pictographs
        ]
        
        for pattern in emoji_patterns:
            clean = re.sub(pattern, '', clean)
        
        # Remove markdown formatting
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)  # Bold
        clean = re.sub(r'\*(.+?)\*', r'\1', clean)      # Italic
        clean = re.sub(r'`(.+?)`', r'\1', clean)        # Code
        
        # Remove "Face:" patterns when followed by nothing or just whitespace
        clean = re.sub(r'(Happy|Sad|Angry|Fear|Surprised?)\s*Face:\s*$', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'(Happy|Sad|Angry|Fear|Surprised?)\s*Face:\s*(?=\n|$)', '', clean)
        
        # Remove extra whitespace
        clean = ' '.join(clean.split())
        
        return clean.strip()
