from typing import Optional, Sequence
import argparse
import sys
import dateparser
from datetime import datetime, timedelta
from dateutil import tz
import feedparser
import requests
import base64
import subprocess
from pathlib import Path
from tqdm.auto import tqdm
from functools import cache
import json
from textwrap import dedent
import re

MATH_RE = re.compile(r"\$(?P<content>([^\$]|(?<=\\)\$)*)\$")



# ARXIV_URL = "http://export.arxiv.org/api/query?search_query=(%(categories)s) AND lastUpdatedDate:[%(start)s TO %(end)s]&max_results=800"
ARXIV_URL = "http://export.arxiv.org/api/query?search_query=(%(categories)s) AND au:cadiou&max_results=2"

logger = None
def setup_logger():
    global logger
    import sys
    import logging
    logger = logging.getLogger(__name__)
    stream = sys.stderr
    sh = logging.StreamHandler(stream=stream)
    logger.addHandler(sh)

def bake_URL(categories: list[str], start: str, end: str) -> str:
    return ARXIV_URL % dict(categories=" OR ".join(categories), start=start, end=end)

@cache
def get_auth_info():
        return subprocess.check_output([
            "gcloud",
            "auth",
            "application-default",
            "print-access-token"
        ]).decode().strip()

def tts(text: str, path: str):
    data = {
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "pitch": 0,
            "effectsProfileId": [
                "headphone-class-device"
            ],
            "speakingRate": 1.1
        },
        "input": {
            "ssml": f"<speak>{text}</speak>",
        },
        "voice": {
            "languageCode": "en-GB",
            "name": "en-GB-Wavenet-F"
        }
    }

    # print(get_auth_info())
    # import IPython; IPython.embed()
    r = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        data=json.dumps(data),
        headers={
            "Authorization": f'Bearer "{get_auth_info()}"',
            "Content-Type": "application/json; charset=utf-8",
        }
    )
    if not r.status_code == 200:
        raise Exception(r.text)

    wav_file = base64.b64decode(r.json()["audioContent"])
    with open(path, "bw") as f:
        f.write(wav_file)

def clean_math(match) -> str:
    txt = match.group("content")
    return (
        txt
        .replace("\\sim", " approximately ")
        .replace("\\approx", " approximately ")
        .replace(">", " greater than")
        .replace("<", " lower than")
        .replace("\geq", " greater or equal to ")
        .replace("\leq", " lower or equal to ")
        .replace("\\", " ")
    )

def clean_str(text: str) -> str:
    """Clean a text from any math strings."""
    # Find all math expressions and replace them
    return MATH_RE.subn(clean_math, text)[0]

def main(argv: Optional[Sequence[str]]=None) -> int:
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output folder", type=str, default="output")
    parser.add_argument("-d", "--date", help="Date to query. Defaults to today.", default=None, type=str)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    if args.verbose:
        logger.setLevel(10)

    categories = ["cat:astro-ph.GA", "cat:astro-ph.CO"]

    # ArXiV papers published on day X have been submitted between 20:00 Eastern US on day X-2 and 19:59 on day X-1
    eastern_US_tz = tz.gettz("US/Eastern")
    local_tz = tz.tzlocal()

    if args.date is None:
        date = datetime.now(tz=local_tz).astimezone(eastern_US_tz)
    else:
        date = dateparser.parse(args.date).astimezone(eastern_US_tz)
        if date.tzname() is None:
            date = date.replace(tzinfo=local_tz)

    offset = timedelta(max(1, (date.weekday() + 6) % 7 - 3))
    end_date = date - offset
    offset = timedelta(max(1, (end_date.weekday() + 6) % 7 - 3))
    start_date = end_date - offset

    start = start_date.strftime("%Y%m%d2000")
    end = end_date.strftime("%Y%m%d1959")

    logger.info("Querying ADS")
    url = bake_URL(categories, start, end)
    r = requests.get(url)
    feed = feedparser.parse(r.text)

    logger.info("Found %s abstracts", len(feed["entries"]))
    output_folder = Path(args.output)
    output_folder.mkdir(exist_ok=True)
    for i, entry in enumerate(tqdm(feed["entries"])):
        title = entry["title"].replace("\n", " ")
        normalized_title = title.replace(" ", "_")
        logger.debug("Processing %s", normalized_title)
        if len(entry["authors"]) > 1:
            author_str = entry["author"] + " et al. "
        else:
            author_str = entry["author"] + ". "

        abstract = clean_str(entry["summary"].replace("\n", " "))
        feed_as_txt = dedent(f"""
        <p>{title}, <break time="200ms"/> by {author_str}</p>

        <p>
        {abstract}.
        </p>
        <p><emphasis level=\"strong\">This</emphasis> was {title} by {author_str}.</p>
        """)
        path = output_folder / f"{i+1}.{normalized_title}.wav"

        print(feed_as_txt)
        tts(feed_as_txt, path)
        break

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))