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


class VTubeSimple(Extension):
    """Simple VTube integration that just works"""
    
    # Class-level tracking to persist across instances
    _sent_messages = set()
    _last_clear_time = time.time()
    _pending_response = ""
    _last_update_time = 0
    _send_task = None
    
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
            # Wait 0.5 seconds to see if more text comes
            await asyncio.sleep(0.5)
            
            # Check if this is still the latest response
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
            # Add emotion
            emotion = self._get_emotion(response_text)
            text_with_emotion = f"{emotion} {response_text}"
            
            # Log what we're doing
            self.agent.context.log.log(
                type="info",
                heading="VTube",
                content=f"Sending: {text_with_emotion[:60]}..."
            )
            
            # Generate audio
            audio_data = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_audio, response_text
            )
            if not audio_data:
                return
                
            # Simple volume pattern
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
                    "duration": len(volumes) * 0.02,
                    "name": "Agent Zero"
                }
            }
            
            response = requests.post(
                "http://192.168.50.67:12393/api/external_audio",
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
        """Enhanced emotion detection"""
        text_lower = text.lower()
        
        # First check if user explicitly requested an emotion
        if "sad" in text_lower and ("face" in text_lower or "emotion" in text_lower or "show" in text_lower):
            return "[sadness]"
        elif "happy" in text_lower and ("face" in text_lower or "emotion" in text_lower or "show" in text_lower):
            return "[joy]"
        elif "angry" in text_lower and ("face" in text_lower or "emotion" in text_lower or "show" in text_lower):
            return "[anger]"
        elif "surprised" in text_lower and ("face" in text_lower or "emotion" in text_lower or "show" in text_lower):
            return "[surprise]"
        elif "fear" in text_lower and ("face" in text_lower or "emotion" in text_lower or "show" in text_lower):
            return "[fear]"
        
        # Check for emotional content in the response
        if any(w in text_lower for w in ["hello", "hi", "happy", "great", "wonderful", "joke", "😊", "😃"]):
            return "[joy]"
        elif any(w in text_lower for w in ["sorry", "sad", "unfortunately", "😢", "😞"]):
            return "[sadness]"
        elif any(w in text_lower for w in ["error", "wrong", "failed", "😠", "😡"]):
            return "[anger]"
        elif any(w in text_lower for w in ["wow", "amazing", "incredible", "😲", "😮"]):
            return "[surprise]"
        elif any(w in text_lower for w in ["scared", "afraid", "fear", "😨", "😰"]):
            return "[fear]"
        else:
            return "[neutral]"
    
    def _generate_audio(self, text):
        """Generate audio with Kokoro TTS"""
        try:
            # Clean text for TTS - remove emojis and markdown
            clean_text = self._clean_text_for_tts(text)
            
            client = OpenAI(
                api_key="not-needed",
                base_url="http://192.168.50.60:8880/v1"
            )
            
            response = client.audio.speech.create(
                model="kokoro",
                voice="af_sky+af_bella",
                response_format="wav",
                input=clean_text
            )
            
            return response.content
            
        except:
            return None
    
    def _clean_text_for_tts(self, text):
        """Remove emojis, markdown, and clean text for TTS"""
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
        
        import re
        clean = text
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