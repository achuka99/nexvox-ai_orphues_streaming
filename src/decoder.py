from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import os
import time
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

torch.set_num_threads(1)

# Initialize SNAC model
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(snac_device)
if snac_device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Warm up the model with a dummy inference
    dummy_codes = [
        torch.randint(0, 4096, (1, 1), dtype=torch.int32, device=snac_device),
        torch.randint(0, 4096, (1, 2), dtype=torch.int32, device=snac_device),
        torch.randint(0, 4096, (1, 4), dtype=torch.int32, device=snac_device)
    ]
    with torch.inference_mode():
        _ = model.decode(dummy_codes)

# Load decoder configuration from environment variables
CUSTOM_TOKEN_PREFIX = "<custom_token_"
CUSTOM_TOKEN_SUFFIX = ">"
MIN_FRAMES_FIRST = int(os.environ["MIN_FRAMES_FIRST"])
MIN_FRAMES_SUBSEQ = int(os.environ["MIN_FRAMES_SUBSEQ"])
PROCESS_EVERY = int(os.environ["PROCESS_EVERY"])

# Audio token configuration
CODE_START_TOKEN_ID = 128257  # Marks the beginning of audio tokens
CODE_TOKEN_OFFSET = 128266    # Minimum valid token ID for audio
STREAM_CHUNK_SIZE_GROUPS = 30  # Number of token groups (7 tokens each) for processing
INITIAL_CHUNK_SIZE_GROUPS = 5  # Smaller initial chunk for faster first audio

def turn_token_into_id(token_string, index):
    """Convert a custom token string to its numeric ID.

    Args:
        token_string (str): The literal token text coming from the model.
        index (int): Absolute token position (used for offset calculation).

    Returns:
        Optional[int]: Numeric token ID or ``None`` if the token is invalid.
    """
    token_string = token_string.strip()
    mod = index % 7

    try:
        digits_str = token_string.removeprefix(CUSTOM_TOKEN_PREFIX).removesuffix(CUSTOM_TOKEN_SUFFIX)
        token_id = int(digits_str) - 10 - (mod * 4096)
        return token_id
    except (ValueError, TypeError):
        return None


def convert_to_audio(multiframe, count):
    """
    Highly optimized version of convert_to_audio that eliminates inefficient 
    tensor operations and reduces CPU-GPU transfers for much faster inference
    on high-end GPUs. Optimized for concurrent requests.
    """
    if len(multiframe) < 7:
        return None
    
    num_frames = len(multiframe) // 7
    
    # Pre-allocate tensors with the right shape and directly on target device
    codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
    codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
    codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
    
    # Fill tensors with direct indexing
    for i in range(num_frames):
        base_idx = i * 7
        codes_0[0, i] = multiframe[base_idx]
        
        codes_1[0, i*2] = multiframe[base_idx + 1]
        codes_1[0, i*2 + 1] = multiframe[base_idx + 4]
        
        codes_2[0, i*4] = multiframe[base_idx + 2]
        codes_2[0, i*4 + 1] = multiframe[base_idx + 3]
        codes_2[0, i*4 + 2] = multiframe[base_idx + 5]
        codes_2[0, i*4 + 3] = multiframe[base_idx + 6]
    
    # validation for range check
    if (torch.any(codes_0 < 0) or torch.any(codes_0 > 4096) or
        torch.any(codes_1 < 0) or torch.any(codes_1 > 4096) or
        torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)):
        return None
    
    codes = [codes_0, codes_1, codes_2]
    
    with torch.inference_mode():   
        audio_hat = model.decode(codes)
        audio_slice = audio_hat[:, :, 2048:4096]
        
        if snac_device == "cuda":
            audio_int16_tensor = (audio_slice * 32767.0).round().to(torch.int16)
            return audio_int16_tensor.cpu().numpy().tobytes()
        else:
            audio_np = audio_slice.numpy()
            return (audio_np * 32767.0).round().astype(np.int16).tobytes()


def apply_fade(audio_data: bytes, sample_rate: int = 24000, fade_duration_ms: float = 10.0) -> bytes:
    """Apply fade-in and fade-out to audio data to prevent clicks/pops.
    
    Args:
        audio_data: Raw audio data in bytes (int16 format)
        sample_rate: Sample rate in Hz (default: 24000)
        fade_duration_ms: Duration of fade in/out in milliseconds (default: 10ms)
        
    Returns:
        Faded audio data in bytes (int16 format)
    """
    # Convert bytes to numpy array of int16
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    num_samples = len(audio_array)
    
    # Calculate number of samples for fade
    fade_samples = int((fade_duration_ms / 1000.0) * sample_rate)
    fade_samples = min(fade_samples, num_samples // 2)  # Ensure we don't overlap fades
    
    if fade_samples == 0 or num_samples < 2 * fade_samples:
        return audio_data  # Not enough samples to apply fade
    
    # Create fade in/out curves (linear for simplicity, can be changed to other curves)
    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
    
    # Apply fade in to first samples
    audio_array[:fade_samples] = (audio_array[:fade_samples].astype(np.float32) * fade_in).astype(np.int16)
    # Apply fade out to last samples
    audio_array[-fade_samples:] = (audio_array[-fade_samples:].astype(np.float32) * fade_out).astype(np.int16)
    
    return audio_array.tobytes()


async def tokens_decoder(token_gen):
    """Decode tokens into audio chunks with special token handling and reduced latency.

    This decoder processes tokens with the following features:
    - Looks for CODE_START_TOKEN_ID to identify the beginning of audio tokens
    - Validates tokens are within the expected range (>= CODE_TOKEN_OFFSET)
    - Uses configurable chunk sizes for initial and subsequent processing
    - Applies fade-in/out to prevent clicks/pops between chunks
    """
    buffer = []
    token_buffer = []
    count = 0
    first_chunk_sent = False
    sample_rate = 24000  # SNAC model sample rate
    in_audio_sequence = False
    
    # Calculate token counts for processing
    initial_chunk_size = INITIAL_CHUNK_SIZE_GROUPS * 7  # 5 groups of 7 tokens
    stream_chunk_size = STREAM_CHUNK_SIZE_GROUPS * 7    # 30 groups of 7 tokens

    async for token_sim in token_gen:
        # Convert token string to ID
        token = turn_token_into_id(token_sim, count)
        if token is None:
            continue
            
        # Check for start of audio sequence
        if token == CODE_START_TOKEN_ID and not in_audio_sequence:
            in_audio_sequence = True
            token_buffer = []
            continue
            
        # If we're not in an audio sequence, skip processing
        if not in_audio_sequence:
            continue
            
        # Validate token is within expected range for audio
        if token < CODE_TOKEN_OFFSET:
            continue
            
        # Add valid audio token to buffer
        token_buffer.append(token)
        count += 1
        
        # Determine if we have enough tokens to process
        tokens_needed = initial_chunk_size if not first_chunk_sent else stream_chunk_size
        
        if len(token_buffer) >= tokens_needed:
            # Process the tokens in groups of 7 (one audio frame)
            num_frames = len(token_buffer) // 7
            if num_frames > 0:
                # Take complete frames (multiples of 7)
                frames_to_process = token_buffer[:num_frames * 7]
                token_buffer = token_buffer[num_frames * 7:]
                
                # Convert to audio
                audio = convert_to_audio(frames_to_process, count)
                if audio is not None:
                    audio = apply_fade(audio, sample_rate)
                    first_chunk_sent = True
                    yield audio