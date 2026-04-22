"""
test_sample_compare.py
Upload test_sample.mp3 to the local server, run all 3 models, compute WER vs GT, show ranking.
Usage:  python test_sample_compare.py
"""
import io, json, re, sys, time
import urllib.request, urllib.parse, urllib.error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SERVER   = "http://localhost:8080"
AUDIO    = r"C:\Users\abhis\Downloads\test_sample.mp3"
MODELS   = [
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3.5",
    "parakeet-tdt-0.6b-v3",
]

GT_TEXT = """
Yeah, it's a van, but it's a, uh, it's a, uh, it's not an MPV, but it's a little van.
Nobody ever showed any interest in it. I think what's happened is that and I knew this would happen.
I knew I'd be embarrassed. That's what I told you. See, when I told, I told you back, I will be positive,
like 100% serious. Jordan, did this veto ban get a price yesterday? Why the fuck do you repeat saying this?
You got a price? What? I think, I think the, the Be like, you guys gave me 9.5 yesterday, give me 9.9,
I'll cut right now and I'll go. What business is intensive? We don't take these sh I don't know, you can
find it as well. Oh yeah, what you saying? Rafiq has come up to me telling me, uh, telling me I'm reviving
a detail van. He's saying if you can get 300 extra on the part exchange, he'll buy right now. I'm saying how?
There is no price, we don't take this. Full service issue. We need more on it because it's got, it's got 26K
out. He say he's talking about 24. I don't think so. By the time I bet, what do we get right now? Check
AutoTrader to see what it says. Green lights. What's going on? That's the best suit. Me? I didn't.
If you do that, I'll drop you.
I'm sorry, I'm sorry. My team picked up the phone. My team picked up the phone, hung up on my face twice.
Hello? I didn't even know. And she said she can't hear you. I was at some, like, I was at the meeting of
some DJ Bombardi. Who's number ending in 5461? And I got him for free.
I need, I need you to transfer it. Let me just take that for you.
Yeah, where's my headset? One second.
Wait, somebody nicked it. Why does it run?
Yeah, just use that one, man. Yeah, no, but that's not right. Open XB10. This is the mouse.
Just connect it down there, man. Okay, open X please in the Chrome. There.
Did you need me yesterday?
Yes, that's good. I was, I was in bed somewhere laying down.
You wouldn't get it.
Um, transport when it comes there.
Hello, we're speaking.
One second. Hello, we're speaking. Hello, hello, can you hear me?
Can you push it through, please? Transfer that green button.
Hello, we're speaking. Really missing. Yes. Hi, how you doing?
You okay?
To be honest, bro, look, this 21-year-old take a picture of that.
Oh yeah, yeah, yeah. Um, no, because obviously I need to speak to accounting. My accounting, um, didn't get
back to me. Plus, always obviously got busy. What I'm going to do, uh, I'm going to, I'm going to honor the
4-year warranty. I'm going to give you that. I'm going to give you extra year warranty. Yes, it's going to
be 4 years, not 3 years. All right.
They are hungry.
No, no, no, no, no, it's all good. It will be an invoice.
I'm going to send you an invoice.
Yeah, yeah, but it won't be that that won't be done today. It's more reason being that because obviously the
accounting, because when we put the car to sold I can't access it. The number has been finalized. Obviously
I need to rechange it with accounting.
Okay, so it'll be tomorrow.
Yeah, no, no, no, no, as I told you yesterday, I'm the manager. Yeah, yeah, just remind me tomorrow though,
please. Yeah, I'm here from 9 to 9. Just give me a quick call, remind me so I can speak, because the
accounting is shut. Yeah, because they can't give me a shot. Thank you, you take care, look after yourself, bye-bye.
Most rough fool I've ever seen in my life. Can I get full scope, please? I'm sorry, tell me about it.
All of you stressing here. Yeah, there's like a thousand. I have a prayer group coming in 8 minutes.
She not in today. Tara needs you.
Are you alone today?
Okay, go and help her out.
Help her.
Come on.
Dennis is free, no? Dennis is free. We're not doing free.
Everyone steals everyone's headsets. I don't know what some how dare someone stealing mine? There's plenty.
Yeah, but they're all broken.
That's why they did.
They're not broken.
Um, Dennis, are you going or not?
There's no one here. That needs to go to okay, if he if that's what it is, then we can do it.
If there's not Yeah, literally, and there's no internet at the moment, so literally, you need to paint the canvas first.
I'll speak to Austin and we'll see what Austin got together.
You ready for this one?
So basically, he's got, um, the CLA that he wants to do a part exchange for. He's not ready with the last one.
We need 25K. We're on 22.6K. He's not having the balls. Full grand, no gate, and he's only paid 10,000.
Just pay a weekly fee for that. He doesn't want to pay, right? He doesn't want to pay 10,000 pounds.
How much more? No, he wants 23,000. We know what you need. Just a touch, premium finance.
So, um, what's the registry? Yeah, and get them accepted.
Yes, it's right.
There's literally nothing wrong with the fast, like there's nothing that needs to be done.
Okay, so how much did you hit him with?
22.6?
I've got another Hanspahn that said I've just set him on this.
Yeah, 26.1.
"""


def norm(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def wer(ref, hyp):
    r = norm(ref).split()
    h = norm(hyp).split()
    R, H = len(r), len(h)
    if R == 0:
        return 100.0 if h else 0.0
    prev = list(range(H + 1))
    for i in range(1, R + 1):
        cur = [i] + [0] * H
        for j in range(1, H + 1):
            cur[j] = prev[j-1] if r[i-1] == h[j-1] else 1 + min(prev[j], cur[j-1], prev[j-1])
        prev = cur
    return prev[H] / R * 100


def api(path):
    url = SERVER + path
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def upload_and_run(model):
    print(f"\n[{model}] Uploading test_sample.mp3 ...", flush=True)
    with open(AUDIO, "rb") as f:
        data = f.read()
    params = urllib.parse.urlencode({"filename": "test_sample.mp3", "model": model})
    url = f"{SERVER}/api/upload?{params}"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Length", str(len(data)))
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  Upload error {e.code}: {body}", flush=True)
        return None, 0
    print(f"  Started: {resp}", flush=True)

    # Poll until done
    dots = 0
    while True:
        time.sleep(4)
        try:
            st = api("/api/status")
        except Exception:
            time.sleep(5)
            continue
        stage = st.get("stage", "?")
        msg   = st.get("message", "")
        elapsed = time.time() - t0
        print(f"  [{elapsed:5.0f}s] {stage}: {msg}", flush=True)
        if st.get("done"):
            result_id = st.get("result_id", "")
            proc_time = time.time() - t0
            print(f"  Done in {proc_time:.1f}s  result_id={result_id}", flush=True)
            return result_id, proc_time
        if st.get("error"):
            print(f"  ERROR: {st['error']}", flush=True)
            return None, time.time() - t0


def extract_text(result_id):
    if not result_id:
        return ""
    try:
        data = api(f"/api/call/{urllib.parse.quote(result_id, safe='')}")
    except Exception as e:
        print(f"  fetch error: {e}", flush=True)
        return ""
    segs = data.get("segments", [])
    return " ".join(s.get("text", "").strip() for s in segs if s.get("text", "").strip())


# ── Run each model sequentially ────────────────────────────────────────────────
GT = GT_TEXT.strip()
gt_words = len(norm(GT).split())
print(f"\nGround truth: {gt_words} words after normalization")
print(f"Audio: {AUDIO}  (~30 min)\n")
print("=" * 70)

results = []
for model in MODELS:
    result_id, proc_time = upload_and_run(model)
    if result_id:
        text = extract_text(result_id)
        w = wer(GT, text)
        words = len(norm(text).split())
        results.append({
            "model": model,
            "result_id": result_id,
            "proc_s": proc_time,
            "words": words,
            "wer": w,
        })
        print(f"  WER vs GT: {w:.1f}%  Words: {words}", flush=True)
    else:
        results.append({"model": model, "result_id": None, "proc_s": proc_time, "words": 0, "wer": 999})
    # wait for server to go idle before next model
    time.sleep(3)

# ── Table ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  RESULTS: test_sample.mp3 vs Ground Truth")
print("=" * 80)
print(f"  {'Rank':<5} {'Model':<35} {'Words':>6} {'Proc':>7} {'WER':>8}")
print(f"  {'-'*5} {'-'*35} {'-'*6} {'-'*7} {'-'*8}")
ranked = sorted(results, key=lambda r: r["wer"])
for i, r in enumerate(ranked, 1):
    model  = r["model"]
    words  = r["words"]
    proc   = f"{r['proc_s']:.0f}s"
    w      = f"{r['wer']:.1f}%" if r["wer"] < 999 else "FAILED"
    print(f"  {i:<5} {model:<35} {words:>6} {proc:>7} {w:>8}")
print("=" * 80)
print(f"\n  WINNER: {ranked[0]['model']}  (WER = {ranked[0]['wer']:.1f}%)")
print("=" * 80)
