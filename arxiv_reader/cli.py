import argparse
import base64
import io
import json
import logging
import random
import re
import subprocess
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import arxiv
import dateparser
import eyed3
import pydub
import requests
from dateutil import tz
from feedgen.feed import FeedGenerator
from tqdm.auto import tqdm

MATH_RE = re.compile(r"\$(?P<content>([^\$]|(?<=\\)\$)*)\$")
DATE_FMT = "%d-%m-%Y"
ARXIV_QUERY = "(%(categories)s) AND lastUpdatedDate:[%(start)s TO %(end)s]"

logger = logging.getLogger(__name__)
stream = sys.stderr
sh = logging.StreamHandler(stream=stream)
logger.addHandler(sh)


@lru_cache(None)
def get_auth_info():
    return (
        subprocess.check_output(
            ["gcloud", "auth", "application-default", "print-access-token"]
        )
        .decode()
        .strip()
    )


def random_voices_generator():
    choices = {
        "en-AU": [f"en-AU-Wavenet-{_}" for _ in "ABCD"],
        "en-IN": [f"en-IN-Wavenet-{_}" for _ in "ABCD"],
        "en-GB": [f"en-GB-Wavenet-{_}" for _ in "ABCDF"],
        "en-US": [f"en-US-Wavenet-{_}" for _ in "ABCDEFGHIJ"],
    }
    keys = list(choices.keys())
    prevName = None
    prevLang = None
    while True:
        languageCode = keys[random.randint(0, len(keys) - 1)]
        while languageCode == prevLang:
            languageCode = keys[random.randint(0, len(keys) - 1)]

        voices = choices[languageCode]

        name = voices[random.randint(0, len(voices) - 1)]
        while name == prevName:
            name = voices[random.randint(0, len(voices) - 1)]

        yield {"languageCode": languageCode, "name": name}
        prevName = name
        prevLang = languageCode


voices_it = random_voices_generator()


def tts(text: str, path: Path) -> Path:
    voice = next(voices_it)
    data = {
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "pitch": 0,
            "effectsProfileId": ["headphone-class-device"],
            "speakingRate": 1,
        },
        "input": {
            "ssml": f"<speak>{text}</speak>",
        },
        "voice": voice,
    }

    logger.debug("Submitting to Google TTS with voice %s", voice["name"])
    r = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        data=json.dumps(data),
        headers={
            "Authorization": f'Bearer "{get_auth_info()}"',
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    if not r.status_code == 200:
        raise Exception(r.text)

    wav_data = base64.b64decode(r.json()["audioContent"])

    # Now convert wav to mp3
    logger.debug("Converting to mp3")
    sound = pydub.AudioSegment.from_wav(io.BytesIO(wav_data))
    sound.export(path, format="mp3")

    return path


def clean_math(match) -> str:
    txt = match.group("content")
    txt = (
        txt.replace("\\sim", " approximately ")
        .replace("\\approx", " approximately ")
        .replace(">", " greater than")
        .replace("<", " lower than")
        .replace(r"\geq", " greater or equal to ")
        .replace(r"\leq", " lower or equal to ")
        # Remove formatting
        .replace("\\mathrm", "")
        .replace("\\mathtt", "")
        .replace("\\mathbf", "")
        .replace("\\mathcal", "")
        .replace("\\rm", "")
        .replace("\\it", "")
        .replace("\\cal", "")
        # Remove \ and { }
        .replace("\\", " ")
        .replace("{", "")
        .replace("}", "")
    )

    UNITS_SPELLED_BASE = {
        "pc": "parsec",
        "ly": "light year",
        "m": "meter",
        "y": "year",
    }
    UNITS_SPELLED = {}
    for unit, unit_spelled in UNITS_SPELLED_BASE.items():
        for prefix, prefix_spelled in {
            "m": "milli",
            "c": "centi",
            "k": "kilo",
            "M": "mega",
            "G": "giga",
            "": "",
        }.items():
            UNITS_SPELLED[f"{prefix}{unit}"] = f"{prefix_spelled} {unit_spelled}"
    UNITS_SPELLED["au"] = "astronomical unit"

    for unit, unit_spelled in UNITS_SPELLED.items():
        txt = txt.replace(unit, unit_spelled)
    return txt


def clean_str(text: str) -> str:
    """Clean a text from any math strings."""
    # Find all math expressions and replace them
    return MATH_RE.subn(clean_math, text)[0]


def get_start_end(base_date: Optional[str]) -> Tuple[datetime, datetime, datetime]:
    # ArXiV papers published on day X have been submitted between 20:00 Eastern US on
    # day X-2 and 19:59 on day X-1
    eastern_US_tz = tz.gettz("US/Eastern")
    local_tz = tz.tzlocal()

    date: Optional[datetime]
    if base_date is None:
        date = datetime.now(tz=local_tz).astimezone(eastern_US_tz)
    else:
        date = dateparser.parse(base_date)
        if date is None:
            try:
                date = datetime.strptime(base_date, "%d/%m/%Y")
            except ValueError:
                raise Exception("Could not parse '%s' as a date." % base_date)

        else:
            date = date.astimezone(eastern_US_tz)
        if date.tzname() is None:
            date = date.replace(tzinfo=local_tz)

    offset: Dict[int, Tuple[Optional[int], Optional[int]]] = {
        0: (-3, -2),  # Mon: Thu->Fri
        1: (-3, -1),  # Tue: Fri->Sat
        2: (-2, -1),  # Wed: Mon->Tue
        3: (-2, -1),  # Thu: Tue->Wed
        4: (-2, -1),  # Fri: Wed->Thu
        5: (None, None),  # No paper on Sat
        6: (None, None),  # No paper on Sun
    }
    off1, off2 = (timedelta(_) for _ in offset[date.weekday()])

    start_date = date + off1
    end_date = date + off2

    return date, start_date, end_date


def get_create_output_folder(output: str, date: datetime) -> Path:
    output_folder = Path(output) / f"{date:{DATE_FMT}}"
    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder


class PaperMetadata(NamedTuple):
    title: Optional[str] = None
    authors: Optional[str] = None
    pubdate: Optional[datetime] = None
    url: Optional[str] = None
    abstract: Optional[str] = None


def set_metadata(filename: Path, metadata: PaperMetadata) -> int:
    song = eyed3.load(filename)
    song.tag.title = metadata.title
    song.tag.artist = metadata.authors
    pubdate = metadata.pubdate
    if not pubdate:
        pubdate = datetime.now()
    song.tag.original_release_date = pubdate.strftime("%Y-%m-%d")
    song.tag.comments.set(metadata.url, "arxiv_url")
    song.tag.comments.set(metadata.abstract, "abstract")
    song.tag.save()
    return 0


def get_metadata(filename: Path) -> PaperMetadata:
    song = eyed3.load(filename)
    if not song:
        return PaperMetadata()
    comments = song.tag.comments
    if comments:
        abstract_obj = comments.get("abstract")
        url_obj = comments.get("arxiv_url")
    else:
        abstract_obj = None
        url_obj = None
    return PaperMetadata(
        title=song.tag.title,
        authors=song.tag.artist,
        pubdate=song.tag.original_release_date,
        url=(url_obj.text if url_obj else ""),
        abstract=(abstract_obj.text if abstract_obj else ""),
    )


def pull(args: argparse.Namespace) -> int:
    categories = ["cat:astro-ph.GA", "cat:astro-ph.CO"]

    date, start_date, end_date = get_start_end(args.date)
    output_folder = get_create_output_folder(args.output, date)

    # No papers on Sat/Sun
    if date.weekday() in (5, 6):
        return 0

    start = start_date.strftime("%Y%m%d1400")
    end = end_date.strftime("%Y%m%d1359")

    logger.info(
        f"Querying ADS from {start_date:%d %m %Y 14:00 EDT} "
        f"to {end_date:%d %m %Y 13:59 EDT}"
    )
    q = ARXIV_QUERY % dict(categories=" OR ".join(categories), start=start, end=end)
    search = arxiv.Search(
        query=q,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    entries = [entry for entry in search.results() if entry.updated == entry.published]

    logger.info("Found %s abstracts", len(entries))

    out_code = 0

    for i, entry in enumerate(tqdm(entries)):
        # Get rid of resubmitted articles

        title = entry.title.replace("\n", " ")
        # We replace slashes and backslashes to prevent issues on Linux/Windows
        normalized_title = (
            re.subn(" +", "_", title)[0].replace("/", " ").replace("\\", " ")
        )
        logger.debug("Processing %s", normalized_title)
        if len(entry.authors) > 1:
            author_str = entry.authors[0].name + " et al. "
        else:
            author_str = entry.authors[0].name + ". "

        abstract = clean_str(entry.summary.replace("\n", " "))
        feed_as_txt = dedent(
            f"""
        <p>{title}, <break time="200ms"/> by {author_str}</p>

        <p>
        {abstract}.
        </p>
        <p><emphasis level=\"strong\">This</emphasis> was {title},
        <break time="200ms"/> by {author_str}.</p>
        """
        )
        path = output_folder / f"{i+1}.{normalized_title}.mp3"
        path = tts(feed_as_txt, path)

        out_code |= set_metadata(
            path,
            PaperMetadata(
                title="title",
                authors=author_str,
                pubdate=entry.published,
                url=f"http://arxiv.org/abs/{entry.entry_id}",
                abstract=entry.summary,
            ),
        )

    return out_code


def create_rss_feed(args: argparse.Namespace) -> int:
    output_folder = Path(args.output)

    fg = FeedGenerator()
    fg.load_extension("podcast")
    fg.title("Daily Arxiv Papers")
    fg.description("Daily astrophysics papers on the arxiv.")
    fg.link(href="http://pub.cphyc.me/Science/arxiv/podcast.rss", rel="alternate")

    eastern_US_tz = tz.gettz("US/Eastern")

    max_time = datetime.now() - timedelta(args.max_time)

    for out in sorted(output_folder.glob("*")):
        if out.is_file():
            continue
        date_str = out.name
        dt = datetime.strptime(date_str, DATE_FMT).astimezone(eastern_US_tz)
        logger.info("Creating feed for date %s", date_str)

        for file in sorted(out.glob("*.mp3")):
            title = " ".join(file.name.replace("_", " ").split(".")[1:-1])
            metadata = get_metadata(file)
            pubdate = metadata.pubdate
            if pubdate and pubdate < max_time:
                continue
            title = metadata.title or title
            url = f"http://pub.cphyc.me/Science/arxiv/{date_str}/{file.name}"
            logger.info("Found mp3 file %s", file)

            fe = fg.add_entry()
            fe.id(url)
            fe.pubDate(dt)
            fe.title(f"{date_str} | {title}")
            fe.description(
                f"{title} by {metadata.authors} on {metadata.pubdate}\n"
                "\n"
                f"{metadata.abstract}"
                "\n"
                f"arXiv: {metadata.url}"
            )
            fe.enclosure(url, 0, "audio/mpeg")

    fg.rss_str(pretty=True)
    fg.rss_file(str(output_folder / args.output))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(func=lambda args: parser.print_help())

    subparsers = parser.add_subparsers()
    parser_pull = subparsers.add_parser("pull")
    parser_pull.add_argument(
        "-o", "--output", help="Output folder", type=str, default="output"
    )
    parser_pull.add_argument(
        "-d", "--date", help="Date to query. Defaults to today.", default=None, type=str
    )
    parser_pull.set_defaults(func=pull)

    parser_rss = subparsers.add_parser("rss")
    parser_pull.add_argument(
        "-o", "--output", help="Output RSS file", type=str, default="podcast.xml"
    )

    parser_rss.add_argument(
        "--max-time",
        help="How far back in time should the entry be added (default: %(default)s).",
        default=30,
        type=int,
    )
    parser_rss.set_defaults(func=create_rss_feed)

    args = parser.parse_args(argv)
    if args.verbose:
        logger.setLevel(10)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
