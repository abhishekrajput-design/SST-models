"""
Single-page Deepgram transcription app.
Upload an audio file → get transcription with speaker diarization.

Run:  python deepgram_app.py
Open:  http://localhost:5050
"""
import os
import json
import tempfile
import requests
from flask import Flask, request, jsonify, render_template_string, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload

# Store last uploaded audio so it can be served for playback
_uploaded_audio = {"path": None, "mimetype": None}

DEEPGRAM_API_KEY = "5e15d956092376f42f12713d35f0b13aa72d5993"

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepgram Transcription</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; }
  .container { max-width: 900px; margin: 0 auto; padding: 30px 20px; }
  h1 { text-align: center; margin-bottom: 8px; font-size: 28px; color: #fff; }
  .subtitle { text-align: center; color: #888; margin-bottom: 30px; font-size: 14px; }

  .upload-area {
    border: 2px dashed #333; border-radius: 12px; padding: 40px; text-align: center;
    cursor: pointer; transition: all 0.3s; background: #1a1a2e; margin-bottom: 20px;
  }
  .upload-area:hover, .upload-area.dragover { border-color: #6c63ff; background: #1f1f35; }
  .upload-area p { color: #aaa; margin-top: 10px; }
  .upload-area .icon { font-size: 48px; }

  .file-info { background: #1a1a2e; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; display: none; justify-content: space-between; align-items: center; }
  .file-info .name { color: #6c63ff; font-weight: 600; }
  .file-info .size { color: #888; font-size: 13px; }

  .btn {
    display: block; width: 100%; padding: 14px; border: none; border-radius: 8px;
    font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.3s;
    background: #6c63ff; color: #fff; margin-bottom: 20px;
  }
  .btn:hover { background: #5a52d5; }
  .btn:disabled { background: #333; cursor: not-allowed; color: #666; }

  .progress { display: none; margin-bottom: 20px; }
  .progress-bar { height: 4px; background: #222; border-radius: 2px; overflow: hidden; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #6c63ff, #48c6ef); width: 0%; transition: width 0.3s; }
  .progress-text { text-align: center; margin-top: 8px; color: #888; font-size: 13px; }

  /* Sticky audio player bar */
  .player-bar {
    display: none; position: sticky; top: 0; z-index: 100;
    background: #13132a; border-bottom: 1px solid #2a2a4a;
    padding: 14px 20px; margin: -30px -20px 20px -20px;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }
  .player-row { display: flex; align-items: center; gap: 14px; }
  .play-btn {
    width: 52px; height: 52px; border-radius: 50%; border: none; cursor: pointer;
    background: #6c63ff; color: #fff; font-size: 22px; display: flex; align-items: center;
    justify-content: center; transition: all 0.2s; flex-shrink: 0;
  }
  .play-btn:hover { background: #5a52d5; transform: scale(1.05); }
  .play-btn:active { transform: scale(0.95); }
  .player-track { flex: 1; }
  .player-info { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .player-info .now-playing { font-size: 13px; color: #aaa; }
  .player-info .time-display { font-size: 13px; color: #6c63ff; font-weight: 600; font-variant-numeric: tabular-nums; }
  .seek-bar { width: 100%; height: 8px; border-radius: 4px; appearance: none; -webkit-appearance: none; background: #222; cursor: pointer; outline: none; }
  .seek-bar::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #6c63ff; cursor: pointer; box-shadow: 0 0 6px rgba(108,99,255,0.5); }
  .seek-bar::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: #6c63ff; cursor: pointer; border: none; }

  .error { background: #2d1b1b; border: 1px solid #5c2a2a; color: #ff6b6b; padding: 14px; border-radius: 8px; margin-bottom: 20px; display: none; }

  .results { display: none; }
  .stats { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .stat { flex: 1; min-width: 120px; background: #1a1a2e; border-radius: 8px; padding: 14px; text-align: center; }
  .stat .val { font-size: 22px; font-weight: 700; color: #6c63ff; }
  .stat .label { font-size: 12px; color: #888; margin-top: 4px; }

  .transcript-box { background: #1a1a2e; border-radius: 10px; padding: 20px; max-height: 60vh; overflow-y: auto; scroll-behavior: smooth; }
  .segment {
    margin-bottom: 14px; padding: 12px 16px; border-radius: 8px; background: #12121f;
    border-left: 4px solid #333; cursor: pointer; transition: all 0.25s;
  }
  .segment:hover { background: #1a1a30; }
  .segment.active { background: #1c1c3a; border-left-color: #6c63ff; box-shadow: 0 0 12px rgba(108,99,255,0.15); }
  .segment .meta { font-size: 12px; color: #888; margin-bottom: 6px; display: flex; gap: 12px; align-items: center; }
  .segment .meta .speaker { font-weight: 600; }
  .segment .meta .confidence { color: #4caf50; }
  .segment .text { line-height: 1.8; font-size: 15px; }
  .segment .word { display: inline; padding: 2px 1px; border-radius: 3px; transition: all 0.12s; }
  .segment .word.spoken { color: #ccc; }
  .segment .word.current-word { background: #6c63ff; color: #fff; padding: 2px 4px; border-radius: 4px; box-shadow: 0 0 8px rgba(108,99,255,0.4); }

  .full-text { margin-top: 20px; }
  .full-text h3 { margin-bottom: 10px; color: #aaa; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
  .full-text pre { background: #12121f; padding: 16px; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; line-height: 1.7; font-family: inherit; font-size: 14px; max-height: 300px; overflow-y: auto; }
</style>
</head>
<body>
<div class="container">
  <!-- Sticky player bar (hidden until transcription done) -->
  <div class="player-bar" id="playerBar">
    <div class="player-row">
      <button class="play-btn" id="playBtn" title="Play / Pause">&#9654;</button>
      <div class="player-track">
        <div class="player-info">
          <span class="now-playing" id="nowPlaying">Ready to play</span>
          <span class="time-display"><span id="currentTime">0:00</span> / <span id="totalTime">0:00</span></span>
        </div>
        <input type="range" class="seek-bar" id="seekBar" min="0" max="100" step="0.1" value="0">
      </div>
    </div>
    <audio id="audioEl" preload="auto"></audio>
  </div>

  <h1>Deepgram Transcription</h1>
  <p class="subtitle">Upload audio &rarr; get transcription with speaker diarization (Nova-3)</p>

  <div class="upload-area" id="dropZone">
    <div class="icon">&#127908;</div>
    <p>Drag & drop audio file here or <strong>click to browse</strong></p>
    <p style="font-size:12px; margin-top:6px;">.mp3, .wav, .m4a, .flac, .ogg, .webm</p>
    <input type="file" id="fileInput" accept="audio/*" hidden>
  </div>

  <div class="file-info" id="fileInfo">
    <span class="name" id="fileName"></span>
    <span class="size" id="fileSize"></span>
  </div>

  <button class="btn" id="transcribeBtn" disabled>Transcribe</button>

  <div class="error" id="errorBox"></div>

  <div class="progress" id="progress">
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <div class="progress-text" id="progressText">Uploading...</div>
  </div>

  <div class="results" id="results">
    <div class="stats" id="stats"></div>
    <div class="transcript-box" id="transcriptBox"></div>
    <div class="full-text">
      <h3>Full Text</h3>
      <pre id="fullText"></pre>
    </div>
  </div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const btn = document.getElementById('transcribeBtn');
const audioEl = document.getElementById('audioEl');
const playBtn = document.getElementById('playBtn');
const seekBar = document.getElementById('seekBar');
const currentTimeEl = document.getElementById('currentTime');
const totalTimeEl = document.getElementById('totalTime');
const playerBar = document.getElementById('playerBar');
const nowPlaying = document.getElementById('nowPlaying');

let selectedFile = null;
let segmentsData = [];
let animFrame = null;
let seekDragging = false;

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) setFile(fileInput.files[0]); });

function setFile(f) {
  selectedFile = f;
  document.getElementById('fileName').textContent = f.name;
  document.getElementById('fileSize').textContent = (f.size / 1024 / 1024).toFixed(2) + ' MB';
  fileInfo.style.display = 'flex';
  btn.disabled = false;
  document.getElementById('results').style.display = 'none';
  document.getElementById('errorBox').style.display = 'none';
  playerBar.style.display = 'none';
}

function fmtTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = (seconds % 60).toFixed(2);
  return h > 0 ? h+':'+String(m).padStart(2,'0')+':'+String(s).padStart(5,'0')
               : m+':'+String(s).padStart(5,'0');
}
function fmtShort(seconds) {
  if (isNaN(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m+':'+String(s).padStart(2,'0');
}

const speakerColors = ['#48c6ef','#6c63ff','#ff6b9d','#ffa726','#4caf50','#e040fb','#00bcd4','#ff5252'];

// ── Audio Player ──────────────────────────────────────────────────────────
playBtn.addEventListener('click', () => {
  if (audioEl.paused) audioEl.play(); else audioEl.pause();
});
audioEl.addEventListener('play', () => {
  playBtn.innerHTML = '&#10074;&#10074;';
  nowPlaying.textContent = 'Playing...';
  startHighlightLoop();
});
audioEl.addEventListener('pause', () => {
  playBtn.innerHTML = '&#9654;';
  nowPlaying.textContent = 'Paused';
  stopHighlightLoop();
});
audioEl.addEventListener('ended', () => {
  playBtn.innerHTML = '&#9654;';
  nowPlaying.textContent = 'Finished';
  stopHighlightLoop();
});
audioEl.addEventListener('loadedmetadata', () => {
  seekBar.max = audioEl.duration;
  totalTimeEl.textContent = fmtShort(audioEl.duration);
});
audioEl.addEventListener('timeupdate', () => {
  if (!seekDragging) {
    seekBar.value = audioEl.currentTime;
    currentTimeEl.textContent = fmtShort(audioEl.currentTime);
  }
});
seekBar.addEventListener('mousedown', () => { seekDragging = true; });
seekBar.addEventListener('touchstart', () => { seekDragging = true; });
seekBar.addEventListener('mouseup', () => { seekDragging = false; });
seekBar.addEventListener('touchend', () => { seekDragging = false; });
seekBar.addEventListener('input', () => {
  audioEl.currentTime = parseFloat(seekBar.value);
  currentTimeEl.textContent = fmtShort(audioEl.currentTime);
  highlightAtTime(audioEl.currentTime);
});

// ── Highlight loop ────────────────────────────────────────────────────────
function startHighlightLoop() {
  stopHighlightLoop();
  (function tick() {
    highlightAtTime(audioEl.currentTime);
    animFrame = requestAnimationFrame(tick);
  })();
}
function stopHighlightLoop() {
  if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
}

function highlightAtTime(t) {
  const box = document.getElementById('transcriptBox');
  if (!box) return;

  // Highlight active segment + auto-scroll
  document.querySelectorAll('.segment').forEach(el => {
    const start = parseFloat(el.dataset.start);
    const end = parseFloat(el.dataset.end);
    if (t >= start && t <= end) {
      if (!el.classList.contains('active')) {
        el.classList.add('active');
        // Scroll segment into view inside the transcript box
        const boxRect = box.getBoundingClientRect();
        const elRect = el.getBoundingClientRect();
        if (elRect.top < boxRect.top || elRect.bottom > boxRect.bottom) {
          el.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }
      }
    } else {
      el.classList.remove('active');
    }
  });

  // Highlight individual words
  document.querySelectorAll('.word').forEach(el => {
    const ws = parseFloat(el.dataset.start);
    const we = parseFloat(el.dataset.end);
    if (t >= ws && t < we) {
      el.classList.add('current-word');
      el.classList.add('spoken');
    } else if (t >= we) {
      el.classList.remove('current-word');
      el.classList.add('spoken');
    } else {
      el.classList.remove('current-word');
      el.classList.remove('spoken');
    }
  });
}

// Click segment to seek + play
function seekTo(start) {
  audioEl.currentTime = start;
  if (audioEl.paused) audioEl.play();
}

// ── Transcribe button ─────────────────────────────────────────────────────
btn.addEventListener('click', async () => {
  if (!selectedFile) return;
  const progress = document.getElementById('progress');
  const progressFill = document.getElementById('progressFill');
  const progressText = document.getElementById('progressText');
  const errorBox = document.getElementById('errorBox');

  btn.disabled = true;
  progress.style.display = 'block';
  errorBox.style.display = 'none';
  document.getElementById('results').style.display = 'none';
  playerBar.style.display = 'none';
  progressFill.style.width = '30%';
  progressText.textContent = 'Uploading audio...';

  const formData = new FormData();
  formData.append('audio', selectedFile);

  try {
    progressFill.style.width = '50%';
    progressText.textContent = 'Transcribing with Deepgram Nova-3...';

    const resp = await fetch('/transcribe', { method: 'POST', body: formData });
    progressFill.style.width = '90%';

    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Transcription failed');

    progressFill.style.width = '100%';
    progressText.textContent = 'Done!';
    setTimeout(() => { progress.style.display = 'none'; }, 500);

    segmentsData = data.segments;

    // Load audio from server for playback
    audioEl.src = '/audio?t=' + Date.now();
    audioEl.load();
    nowPlaying.textContent = selectedFile.name;
    playerBar.style.display = 'block';

    renderResults(data);

    // Auto-play after a short delay to let audio load
    audioEl.addEventListener('canplay', function autoPlay() {
      audioEl.removeEventListener('canplay', autoPlay);
      audioEl.play();
    });
  } catch (err) {
    progress.style.display = 'none';
    errorBox.textContent = err.message;
    errorBox.style.display = 'block';
  }
  btn.disabled = false;
});

function renderResults(data) {
  const results = document.getElementById('results');
  const stats = document.getElementById('stats');
  const box = document.getElementById('transcriptBox');
  const fullText = document.getElementById('fullText');

  stats.innerHTML = `
    <div class="stat"><div class="val">${data.segments.length}</div><div class="label">Segments</div></div>
    <div class="stat"><div class="val">${data.speakers}</div><div class="label">Speakers</div></div>
    <div class="stat"><div class="val">${data.duration}</div><div class="label">Duration (s)</div></div>
    <div class="stat"><div class="val">${data.avg_confidence}</div><div class="label">Avg Confidence</div></div>
  `;

  box.innerHTML = data.segments.map((seg, i) => {
    const spkIdx = parseInt(seg.speaker.replace('Speaker ','')) || 0;
    const color = speakerColors[spkIdx % speakerColors.length];

    let textHtml;
    if (seg.words && seg.words.length > 0) {
      textHtml = seg.words.map(w =>
        '<span class="word" data-start="'+w.start+'" data-end="'+w.end+'">'+w.word+' </span>'
      ).join('');
    } else {
      textHtml = '<span class="word" data-start="'+seg.start+'" data-end="'+seg.end+'">'+seg.text+'</span>';
    }

    return '<div class="segment" data-start="'+seg.start+'" data-end="'+seg.end+'" '
      +'style="border-left-color:'+color+'" onclick="seekTo('+seg.start+')">'
      +'<div class="meta">'
      +'<span class="speaker" style="color:'+color+'">'+seg.speaker+'</span>'
      +'<span>'+fmtTime(seg.start)+' &rarr; '+fmtTime(seg.end)+'</span>'
      +'<span class="confidence">'+(seg.confidence*100).toFixed(1)+'%</span>'
      +'</div>'
      +'<div class="text">'+textHtml+'</div>'
      +'</div>';
  }).join('');

  fullText.textContent = data.full_text;
  results.style.display = 'block';
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio = request.files["audio"]
    audio_data = audio.read()

    if not audio_data:
        return jsonify({"error": "Empty file"}), 400

    # Save uploaded audio for playback via /audio endpoint
    ext = os.path.splitext(audio.filename)[1].lower()
    if _uploaded_audio["path"] and os.path.isfile(_uploaded_audio["path"]):
        os.remove(_uploaded_audio["path"])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=tempfile.gettempdir())
    tmp.write(audio_data)
    tmp.close()

    # Determine content type
    content_types = {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".ogg": "audio/ogg", ".webm": "audio/webm",
    }
    content_type = content_types.get(ext, "audio/mpeg")
    _uploaded_audio["path"] = tmp.name
    _uploaded_audio["mimetype"] = content_type

    # Call Deepgram API
    try:
        resp = requests.post(
            "https://api.deepgram.com/v1/listen",
            params={
                "model": "nova-3",
                "language": "en",
                "smart_format": "true",
                "diarize": "true",
                "utterances": "true",
                "punctuate": "true",
            },
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": content_type,
            },
            data=audio_data,
            timeout=300,
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        err_body = resp.text if resp else str(e)
        return jsonify({"error": f"Deepgram API error: {err_body}"}), 502
    except Exception as e:
        return jsonify({"error": f"Request failed: {e}"}), 500

    dg = resp.json()

    # Parse results
    results = dg.get("results", {})
    utterances = results.get("utterances", [])
    channels = results.get("channels", [])

    # Build word-level map from channel words (has precise timestamps)
    all_words = []
    if channels:
        alts = channels[0].get("alternatives", [])
        if alts:
            for w in alts[0].get("words", []):
                all_words.append({
                    "word": w.get("punctuated_word", w.get("word", "")),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "speaker": w.get("speaker", 0),
                })

    # Build segments from utterances (has speaker info) + attach words
    segments = []
    for utt in utterances:
        utt_start = utt["start"]
        utt_end = utt["end"]
        # Find words belonging to this utterance by time overlap
        seg_words = [w for w in all_words if w["start"] >= utt_start - 0.05 and w["end"] <= utt_end + 0.05]
        segments.append({
            "speaker": f"Speaker {utt.get('speaker', '?')}",
            "start": round(utt_start, 2),
            "end": round(utt_end, 2),
            "text": utt.get("transcript", ""),
            "confidence": round(utt.get("confidence", 0), 4),
            "words": seg_words,
        })

    # Full transcript text
    full_text = ""
    if channels:
        alts = channels[0].get("alternatives", [])
        if alts:
            full_text = alts[0].get("transcript", "")

    # Stats
    speakers = len(set(s["speaker"] for s in segments)) if segments else 0
    duration = round(dg.get("metadata", {}).get("duration", 0), 2)
    avg_conf = round(sum(s["confidence"] for s in segments) / len(segments), 4) if segments else 0

    return jsonify({
        "segments": segments,
        "full_text": full_text,
        "speakers": speakers,
        "duration": duration,
        "avg_confidence": avg_conf,
    })


@app.route("/audio")
def serve_audio():
    if not _uploaded_audio["path"] or not os.path.isfile(_uploaded_audio["path"]):
        return "No audio uploaded", 404
    return send_file(_uploaded_audio["path"], mimetype=_uploaded_audio["mimetype"])


if __name__ == "__main__":
    print("Starting Deepgram Transcription App...")
    print("Open http://localhost:5050 in your browser")
    app.run(host="0.0.0.0", port=5050, debug=False)
