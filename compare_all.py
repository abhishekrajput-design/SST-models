"""Compare transcription/diarization accuracy across three sources:
   1. Ground truth (the text block the user pasted)
   2. jebebr.txt (another system's output)
   3. Our system (Distil-Whisper v3.5 + ECAPA diarization)
"""
import json, re, os, io, sys

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Ground truth from the user's pasted text (Speaker A / Speaker B format)
GROUND_TRUTH = """Speaker A
Hello, August.

Speaker B
Hello.

Speaker A
Hello, August, it's Zach calling. How are you?

Speaker B
I'm okay. How are you?

Speaker A
I'm good. I'm good. I'm good. Have you managed to have a look at open banking link we discussed earlier?

Speaker B
Sorry?

Speaker A
Did you find the link I sent to you earlier from Jeannie or no?

Speaker B
I'm sending to him my document.

Speaker A
Okay, you've submitted your documents on Genie? Yeah. Okay, good. Give me 2 seconds, I'm gonna give you an answer now, just to see if they've replied to us. Obviously it was Easter Monday today, so yesterday and today were like the worst days for finance. Yeah. But I'm literally, I'm gonna pop you on hold for 2 minutes and see if I can give you an answer. Alright. Hello, August? Yes, mate?

Speaker B
Yes, I just had a look at the portal, so it's looking good.

Speaker A
They're reviewing your documents now, so obviously because it's Easter, They haven't sent you the finance documents yet, but tomorrow morning they're going to send us the finance documents and I'm going to send them to you. All right.

Speaker B
And can you just, can you send me my, uh, uh, plan list, please?

Speaker A
Plan list?

Speaker B
Yeah. Payment plan.

Speaker A
So your payment tomorrow morning, the finance documents going to send it to us. All right. And it's going to be, um, as I discussed, so like, you know, the very first time, I think it's 270 now. 276? 276. It's 276. So yesterday it was 283, then 300, but now it's gone back down to 276 and they haven't asked for anything so far. They've accepted your open banking. So tomorrow morning they're going to send me these numbers and I'm going to send it to you in your email. Yeah.

Speaker B
It's how many months?

Speaker A
It's how many months? Yeah. The term, how long is the term? Is that what you're asking me? How long is the finance contract? Huh? August. Yeah. It's 270 pounds for 5 years.

Speaker B
5 years?

Speaker A
It's 5 years long, but you can always settle it earlier as we discussed. Like the earlier you settle it, the better for you. Same as we discussed. Is there any issues? Tell me if there's any issues.

Speaker B
How much I will use English?

Speaker A
How much are you financing?

Speaker B
Yeah.

Speaker A
So let me tell you now. So the vehicle price, I'll tell you now, sir, one second.

Speaker B
Because today, you know, brothers

Speaker A
Tell me, August, talk to me.

Speaker B
Yes. Uh, some company is offered to me as well. He sent to me is a pay plan.

Speaker A
Yeah.

Speaker B
Is, uh, you are just applying, uh, cash deposit 3,800 pounds.

Speaker A
On their application or on our application?

Speaker B
Is different.

Speaker A
You, you, you. Are you sure it's from us?

Speaker B
Exactly.

Speaker A
No, so from us, yeah, your deposit, you have two options. So either your deposit is 6,100 pounds.

Speaker B
Yeah.

Speaker A
And you're financing 8,350 pounds.

Speaker B
Yeah.

Speaker A
Yeah. Or you have your other option, which obviously is a bit lower deposit. That we currently done with a 3-year warranty and a 3-year service plan. So the car is actually protected. So you have the option, the option's with you. Obviously at Car Planner we give you that package.

Speaker B
Okay, you send me all the documents tomorrow.

Speaker A
Yeah.

Speaker B
And pay plan.

Speaker A
Of course.

Speaker B
Let me check.

Speaker A
I'll call you in the morning as well when

Speaker B
All right, all right, all right.

Speaker A
When you have a bit of peace of mind. But look, last thing I want to tell you, August, Yeah. Is if I send you a 250 pound payment link, are you able to pay it now? Only reason why I'm asking is because this car has a lot of interest and we have an active application with you. It's fully refundable if you're not happy with the finance. It's just to secure the vehicle. That's it.

Speaker B
Alright, I need is sending to you some deposit today.

Speaker A
Yeah, today's deposit was only 250 pounds just to save the car. No, no, no, no.

Speaker B
Just listen to me. I try, I did try sending to you, but my account is not available. I have the cash. I didn't deposit to my account. Everywhere is closed, you know?

Speaker A
That's why you asked me if I take cash earlier, yes, okay. Yeah. I understand.

Speaker B
If you remember.

Speaker A
I remember, of course, I remember everything you say. No, no, that's true. I remember you are

Speaker B
No worries. Just about the money, you let me know.

Speaker A
Okay. No worries. Everything is okay. Okay, no worries. Only thing I'm saying is obviously the car might sell. You know how we are where we switch to cheapest in 100-mile radius. So obviously if the car sells, there's nothing I can do to protect it for you. That's the only reason why I understand what you're saying. But the only reason why I'm telling you is due to the urgency. That's all it is, okay?

Speaker B
It's okay, it's okay.

Speaker A
No worries. Alright, I'll speak to you tomorrow, August. See you. Bye-bye. Take care."""


def parse_ab_text(text):
    """Parse 'Speaker A\\n...\\n\\nSpeaker B\\n...' format into [(spk, phrase), ...]"""
    turns = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().split("\n", 1)
        if len(lines) < 2: continue
        spk = lines[0].strip()
        if spk not in ("Speaker A", "Speaker B"): continue
        phrase = lines[1].strip()
        turns.append((spk.replace("Speaker ", ""), phrase))
    return turns


def parse_jebebr(path):
    """Parse the jebebr.txt JSON-like transcription list."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # Extract the JSON array
    m = re.search(r'\[(\s*\{.*\})\s*\]', content, re.DOTALL)
    arr = json.loads("[" + m.group(1) + "]")
    turns = []
    for e in arr:
        spk = e.get("speaker", "")
        if "0" in spk: spk = "A"
        elif "1" in spk: spk = "B"
        turns.append((spk, e.get("phrase", "").strip()))
    return turns


def parse_our_system(folder):
    """Parse our system's result.json."""
    with open(os.path.join(folder, "result.json"), encoding="utf-8") as f:
        data = json.load(f)
    tj = data.get("transcription_json") or []
    turns = []
    for e in tj:
        spk = e.get("speaker", "").replace("Speaker ", "")
        turns.append((spk, e.get("phrase", "").strip()))
    return turns, data.get("processing_time_seconds", 0)


def normalize(s):
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wer(ref_words, hyp_words):
    """Word Error Rate via edit distance (lower is better)."""
    R, H = len(ref_words), len(hyp_words)
    if R == 0: return 1.0 if H else 0.0
    prev = list(range(H + 1))
    for i in range(1, R + 1):
        cur = [i] + [0] * H
        for j in range(1, H + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                cur[j] = prev[j-1]
            else:
                cur[j] = 1 + min(prev[j], cur[j-1], prev[j-1])
        prev = cur
    return prev[H] / R


# ---- Load all three ----
gt_turns = parse_ab_text(GROUND_TRUTH)
gt_text = " ".join(t[1] for t in gt_turns)
gt_words = normalize(gt_text).split()

jebebr_turns = parse_jebebr(r"C:\Users\abhis\Downloads\jebebr.txt")
jebebr_text = " ".join(t[1] for t in jebebr_turns)
jebebr_words = normalize(jebebr_text).split()

# Try the Distil-Whisper test folder (best model from earlier test)
CANDIDATES = [
    ("our-system-distil-v3.5",
     "call_processor/data/processed/enhanced_acc_distil_whisper_large_v3.5__distil-whisper-large-v3.5"),
    ("our-system-whisper-v3",
     "call_processor/data/processed/enhanced_acc_whisper_large_v3__whisper-large-v3"),
    ("our-system-turbo",
     "call_processor/data/processed/enhanced_acc_whisper_large_v3_turbo__whisper-large-v3-turbo"),
    ("our-system-parakeet",
     "call_processor/data/processed/enhanced_acc_parakeet_tdt_0.6b_v3__parakeet-tdt-0.6b-v3"),
    ("our-system-deepgram",
     "call_processor/data/processed/enhanced_acc_deepgram_nova_3__deepgram-nova-3"),
]

print(f"Ground truth (pasted): {len(gt_turns)} turns, {len(gt_words)} words")
print(f"jebebr.txt: {len(jebebr_turns)} turns, {len(jebebr_words)} words\n")

# Score jebebr against ground truth
jebebr_wer = wer(gt_words, jebebr_words) * 100
gt_spk_flips  = sum(1 for i in range(1,len(gt_turns))     if gt_turns[i][0]     != gt_turns[i-1][0])
jeb_spk_flips = sum(1 for i in range(1,len(jebebr_turns)) if jebebr_turns[i][0] != jebebr_turns[i-1][0])

print("=" * 85)
print(f"  {'Source':<32} {'Turns':>6} {'Words':>7} {'WER':>7} {'Spk-flips':>10} {'Time':>7}")
print(f"  {'-'*32} {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*7}")
print(f"  {'GROUND TRUTH (pasted)':<32} {len(gt_turns):>6} {len(gt_words):>7} {'0.0%':>7} {gt_spk_flips:>10} {'-':>7}")
print(f"  {'jebebr.txt (other system)':<32} {len(jebebr_turns):>6} {len(jebebr_words):>7} {jebebr_wer:>6.1f}% {jeb_spk_flips:>10} {'-':>7}")

for name, folder in CANDIDATES:
    if not os.path.exists(os.path.join(folder, "result.json")):
        continue
    turns, t_s = parse_our_system(folder)
    text = " ".join(t[1] for t in turns)
    words = normalize(text).split()
    w = wer(gt_words, words) * 100
    flips = sum(1 for i in range(1,len(turns)) if turns[i][0] != turns[i-1][0])
    print(f"  {name:<32} {len(turns):>6} {len(words):>7} {w:>6.1f}% {flips:>10} {t_s:>5.0f}s")

print("=" * 85)
print("\n  LEGEND:")
print("  WER       = Word Error Rate vs ground truth (lower = better; 0% = perfect)")
print("  Spk-flips = Number of speaker changes between adjacent turns")
print("              (should be close to ground truth's 86 flips)")
