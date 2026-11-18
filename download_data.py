import ssl
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import certifi

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
# original SciFact release: has the SUPPORT/CONTRADICT labels that BEIR drops
CLAIMS_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

# certificates stay verified; certifi's CA bundle has the HARICA root some OS stores lack
urllib.request.install_opener(urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))))

Path("data").mkdir(exist_ok=True)
print("Downloading SciFact BEIR (~3 MB)...")
urllib.request.urlretrieve(BEIR_URL, "data/scifact.zip")
with zipfile.ZipFile("data/scifact.zip") as z:
    z.extractall("data")

print("Downloading SciFact claims + labels (~3 MB)...")
urllib.request.urlretrieve(CLAIMS_URL, "data/scifact_claims.tar.gz")
out = Path("data/scifact_claims")
out.mkdir(exist_ok=True)
with tarfile.open("data/scifact_claims.tar.gz") as t:
    for name in ("claims_train.jsonl", "claims_dev.jsonl"):
        member = t.getmember(f"data/{name}")
        member.name = name
        t.extract(member, out, filter="data")
print("Done: data/scifact/ and data/scifact_claims/")
