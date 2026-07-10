import urllib.request
import zipfile
from pathlib import Path
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

Path("data").mkdir(exist_ok=True)
print("Downloading SciFact (~3 MB)...")
urllib.request.urlretrieve(URL, "data/scifact.zip")
with zipfile.ZipFile("data/scifact.zip") as z:
    z.extractall("data")
print("Done. Extracted to data/scifact/")
