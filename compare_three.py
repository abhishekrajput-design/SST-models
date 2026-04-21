"""Three-way comparison: Ground Truth vs jebebr.txt vs Our System
Outputs side-by-side table with alignment and accuracy stats.
"""
import json, re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Ground Truth (user-pasted) ───────────────────────────────────────
GT = [
    ("A","Hello, August."),
    ("B","Hello."),
    ("A","Hello, August, it's Zach calling. How are you?"),
    ("B","I'm okay. How are you?"),
    ("A","I'm good. I'm good. I'm good. Have you managed to have a look at open banking link we discussed earlier?"),
    ("B","Sorry?"),
    ("A","Did you find the link I sent to you earlier from Jeannie or no?"),
    ("B","I'm sending to him my document."),
    ("A","Okay, you've submitted your documents on Genie? Yeah. Okay, good. Give me 2 seconds, I'm gonna give you an answer now, just to see if they've replied to us. Obviously it was Easter Monday today, so yesterday and today were like the worst days for finance. Yeah. But I'm literally, I'm gonna pop you on hold for 2 minutes and see if I can give you an answer. Alright. Hello, August? Yes, mate?"),
    ("B","Yes, I just had a look at the portal, so it's looking good."),
    ("A","They're reviewing your documents now, so obviously because it's Easter, They haven't sent you the finance documents yet, but tomorrow morning they're going to send us the finance documents and I'm going to send them to you. All right."),
    ("B","And can you just, can you send me my, uh, uh, plan list, please?"),
    ("A","Plan list?"),
    ("B","Yeah. Payment plan."),
    ("A","So your payment tomorrow morning, the finance documents going to send it to us. All right. And it's going to be, um, as I discussed, so like, you know, the very first time, I think it's 270 now. 276? 276. It's 276. So yesterday it was 283, then 300, but now it's gone back down to 276 and they haven't asked for anything so far. They've accepted your open banking. So tomorrow morning they're going to send me these numbers and I'm going to send it to you in your email. Yeah."),
    ("B","It's how many months?"),
    ("A","It's how many months? Yeah. The term, how long is the term? Is that what you're asking me? How long is the finance contract? Huh? August. Yeah. It's 270 for 5 years."),
    ("B","5 years?"),
    ("A","It's 5 years long, but you can always settle it earlier as we discussed. Like the earlier you settle it, the better for you. Same as we discussed. Is there any issues? Tell me if there's any issues."),
    ("B","How much I will use English?"),
    ("A","How much are you financing?"),
    ("B","Yeah."),
    ("A","So let me tell you now. So the vehicle price, I'll tell you now, sir, one second."),
    ("B","Because today, you know, brothers"),
    ("A","Tell me, August, talk to me."),
    ("B","Yes. Uh, some company is offered to me as well. He sent to me is a pay plan."),
    ("A","Yeah."),
    ("B","Is, uh, you are just applying, uh, cash deposit 3,800."),
    ("A","On their application or on our application?"),
    ("B","Is different."),
    ("A","You, you, you. Are you sure it's from us?"),
    ("B","Exactly."),
    ("A","No, so from us, yeah, your deposit, you have two options. So either your deposit is 6,100."),
    ("B","Yeah."),
    ("A","And you're financing 8,350."),
    ("B","Yeah."),
    ("A","Yeah. Or you have your other option, which obviously is a bit lower deposit. That we currently done with a 3-year warranty and a 3-year service plan. So the car is actually protected. So you have the option, the option's with you. Obviously at Car Planner we give you that package."),
    ("B","Okay, you send me all the documents tomorrow."),
    ("A","Yeah."),
    ("B","And pay plan."),
    ("A","Of course."),
    ("B","Let me check."),
    ("A","I'll call you in the morning as well when"),
    ("B","All right, all right, all right."),
    ("A","When you have a bit of peace of mind. But look, last thing I want to tell you, August, Yeah. Is if I send you a 250 payment link, are you able to pay it now? Only reason why I'm asking is because this car has a lot of interest and we have an active application with you. It's fully refundable if you're not happy with the finance. It's just to secure the vehicle. That's it."),
    ("B","Alright, I need is sending to you some deposit today."),
    ("A","Yeah, today's deposit was only 250 just to save the car. No, no, no, no."),
    ("B","Just listen to me. I try, I did try sending to you, but my account is not available. I have the cash. I didn't deposit to my account. Everywhere is closed, you know?"),
    ("A","That's why you asked me if I take cash earlier, yes, okay. Yeah. I understand."),
    ("B","If you remember."),
    ("A","I remember, of course, I remember everything you say. No, no, that's true. I remember you are"),
    ("B","No worries. Just about the money, you let me know."),
    ("A","Okay. No worries. Everything is okay. Okay, no worries. Only thing I'm saying is obviously the car might sell. You know how we are where we switch to cheapest in 100-mile radius. So obviously if the car sells, there's nothing I can do to protect it for you. That's the only reason why I understand what you're saying. But the only reason why I'm telling you is due to the urgency. That's all it is, okay?"),
    ("B","It's okay, it's okay."),
    ("A","No worries. Alright, I'll speak to you tomorrow, August. See you. Bye-bye. Take care."),
]

# ── Load jebebr.txt ──────────────────────────────────────────────────
with open(r"C:\Users\abhis\Downloads\jebebr.txt", encoding="utf-8") as f:
    content = f.read()
m = re.search(r'\[(\s*\{.*\})\s*\]', content, re.DOTALL)
jeb_raw = json.loads("[" + m.group(1) + "]")
jebebr = [("A" if "0" in e["speaker"] else "B", e["phrase"]) for e in jeb_raw]

# ── Load our system ───────────────────────────────────────────────────
with open(r"C:\Users\abhis\Downloads\20260406T195131926_406204_transcript.txt", encoding="utf-8") as f:
    ours_text = f.read()
ours = []
lines = ours_text.split("\n")
i = 0
while i < len(lines):
    m = re.match(r"(Speaker [AB])\s+\[", lines[i].strip())
    if m and i+1 < len(lines):
        spk = m.group(1).replace("Speaker ","")
        phrase = lines[i+1].strip()
        if phrase:
            ours.append((spk, phrase))
        i += 2
    else:
        i += 1

# ── Utility ──────────────────────────────────────────────────────────
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()

def truncate(s, n):
    return s if len(s) <= n else s[:n-3] + "..."

def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    R, H = len(r), len(h)
    if R == 0: return 100.0 if h else 0.0
    prev = list(range(H+1))
    for i in range(1, R+1):
        cur = [i] + [0]*H
        for j in range(1, H+1):
            cur[j] = prev[j-1] if r[i-1]==h[j-1] else 1+min(prev[j], cur[j-1], prev[j-1])
        prev = cur
    return prev[H] / R * 100

# Full-text WER
gt_full  = " ".join(p for _,p in GT)
jeb_full = " ".join(p for _,p in jebebr)
our_full = " ".join(p for _,p in ours)

print("\n" + "="*100)
print("  SOURCE STATS")
print("="*100)
print(f"  {'Source':<25} {'Turns':>6} {'Words':>7} {'Speakers':>10} {'WER vs GT':>12}")
print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*10} {'-'*12}")
print(f"  {'Ground Truth (pasted)':<25} {len(GT):>6} {len(gt_full.split()):>7} {'A, B':>10} {'0.0%':>12}")
print(f"  {'jebebr.txt':<25} {len(jebebr):>6} {len(jeb_full.split()):>7} {'A, B':>10} {wer(gt_full,jeb_full):>11.1f}%")
print(f"  {'Our System (distil)':<25} {len(ours):>6} {len(our_full.split()):>7} {'A, B':>10} {wer(gt_full,our_full):>11.1f}%")
print("="*100)

# Side-by-side comparison — best alignment by content similarity
print("\n" + "="*130)
print("  SIDE-BY-SIDE SAMPLE (first 15 ground-truth turns)")
print("="*130)
print(f"  {'#':<3} {'Spk':<3} {'Ground Truth':<40} {'jebebr.txt':<40} {'Our System':<40}")
print(f"  {'-'*3} {'-'*3} {'-'*40} {'-'*40} {'-'*40}")

# Simple sequential alignment (limited to first 15 for readability)
for i in range(min(15, len(GT))):
    spk, gt_phrase = GT[i]
    # Find matching phrase in jebebr by content overlap
    gt_words = set(norm(gt_phrase).split())
    jeb_best = ("", 0)
    for js, jp in jebebr:
        jp_words = set(norm(jp).split())
        overlap = len(gt_words & jp_words)
        if overlap > jeb_best[1]:
            jeb_best = (jp, overlap)
    our_best = ("", 0)
    for os, op in ours:
        op_words = set(norm(op).split())
        overlap = len(gt_words & op_words)
        if overlap > our_best[1]:
            our_best = (op, overlap)
    print(f"  {i+1:<3} {spk:<3} {truncate(gt_phrase,38):<40} {truncate(jeb_best[0],38):<40} {truncate(our_best[0],38):<40}")

print("="*130)

# Turn-level speaker accuracy (rough - by word overlap alignment)
def align_and_check_spk(gt, sys):
    matches = 0
    for gt_spk, gt_phrase in gt:
        gt_words = set(norm(gt_phrase).split())
        best_spk, best_overlap = None, 0
        for s_spk, s_phrase in sys:
            s_words = set(norm(s_phrase).split())
            overlap = len(gt_words & s_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = s_spk
        if best_spk == gt_spk:
            matches += 1
    return matches, len(gt)

j_match, j_total = align_and_check_spk(GT, jebebr)
o_match, o_total = align_and_check_spk(GT, ours)

# Note: speakers might be swapped — try both orderings
o_match_swapped, _ = align_and_check_spk(GT, [("B" if s=="A" else "A", p) for s,p in ours])
j_match_swapped, _ = align_and_check_spk(GT, [("B" if s=="A" else "A", p) for s,p in jebebr])
o_best = max(o_match, o_match_swapped)
j_best = max(j_match, j_match_swapped)

print("\n" + "="*100)
print("  DIARIZATION ACCURACY (speaker label correctness)")
print("="*100)
print(f"  {'Source':<25} {'Correct speaker':>20} {'Accuracy':>12}")
print(f"  {'-'*25} {'-'*20} {'-'*12}")
print(f"  {'jebebr.txt':<25} {f'{j_best}/{j_total}':>20} {100*j_best/j_total:>11.1f}%")
print(f"  {'Our System':<25} {f'{o_best}/{o_total}':>20} {100*o_best/o_total:>11.1f}%")
print("="*100)

# Final winner
print("\n" + "="*100)
print("  FINAL VERDICT")
print("="*100)
our_wer = wer(gt_full, our_full)
jeb_wer = wer(gt_full, jeb_full)
winner = "Our System" if our_wer < jeb_wer else "jebebr.txt"
print(f"  Lower WER wins:  {winner}")
print(f"  Our WER = {our_wer:.1f}%   jebebr WER = {jeb_wer:.1f}%   Diff = {abs(our_wer-jeb_wer):.1f}pp")
print(f"  Our speaker accuracy: {100*o_best/o_total:.1f}%   jebebr: {100*j_best/j_total:.1f}%")
print("="*100)
