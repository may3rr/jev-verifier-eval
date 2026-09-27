"""Parallel range download of Qwen3.5-9B from hf-mirror and ModelScope at once."""
import hashlib, os, queue, subprocess, threading, time, requests
DST = "/root/models/Qwen3.5-9B"
HF = "https://hf-mirror.com/Qwen/Qwen3.5-9B/resolve/main/"
MS = "https://www.modelscope.cn/models/Qwen/Qwen3.5-9B/resolve/master/"
SMALL = ['chat_template.jinja', 'config.json', 'merges.txt', 'model.safetensors.index.json', 'preprocessor_config.json',
         'tokenizer.json', 'tokenizer_config.json', 'video_preprocessor_config.json', 'vocab.json']
SHARDS = [f"model.safetensors-0000{i}-of-00004.safetensors" for i in range(1, 5)]
CHUNK = 32 << 20
os.makedirs(DST, exist_ok=True)
for f in SMALL:
    subprocess.run(["curl", "-sfL", "-o", f"{DST}/{f}", HF + f], check=True)
meta = {}
for f in SHARDS:
    h = requests.head(HF + f, allow_redirects=False, timeout=30).headers
    meta[f] = (int(h.get("x-linked-size") or requests.head(HF + f, allow_redirects=True).headers["content-length"]),
               h.get("x-linked-etag", "").strip('"'))
    with open(f"{DST}/{f}", "wb") as fh:
        fh.truncate(meta[f][0])
q = queue.Queue()
for f in SHARDS:
    for start in range(0, meta[f][0], CHUNK):
        q.put((f, start, min(start + CHUNK, meta[f][0]) - 1))
total = sum(m[0] for m in meta.values()); done = [0]; lock = threading.Lock()

def worker(base):
    while True:
        try:
            f, a, b = q.get_nowait()
        except queue.Empty:
            return
        for attempt in range(8):
            try:
                r = requests.get(base + f, headers={"Range": f"bytes={a}-{b}"}, timeout=60)
                if r.status_code == 206 and len(r.content) == b - a + 1:
                    fd = os.open(f"{DST}/{f}", os.O_WRONLY); os.pwrite(fd, r.content, a); os.close(fd)
                    with lock: done[0] += len(r.content)
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            q.put((f, a, b))

threads = [threading.Thread(target=worker, args=(HF,)) for _ in range(16)] + \
          [threading.Thread(target=worker, args=(MS,)) for _ in range(8)]
for t in threads: t.start()
t0 = time.time()
while any(t.is_alive() for t in threads):
    time.sleep(30)
    print(f"{done[0]/2**30:.2f}/{total/2**30:.2f} GiB  {done[0]/(time.time()-t0)/2**20:.1f} MB/s", flush=True)
for f in SHARDS:
    h = hashlib.sha256()
    with open(f"{DST}/{f}", "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""): h.update(blk)
    print(f, "sha256", "OK" if h.hexdigest() == meta[f][1] else f"MISMATCH {h.hexdigest()} vs {meta[f][1]}", flush=True)
print("DLDONE", flush=True)
