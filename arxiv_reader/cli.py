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
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple
from urllib.parse import quote, unquote

import arxiv
import dateparser
import eyed3
import pydub
import requests
import toml
from dateutil import tz
from feedgen.feed import FeedGenerator
from tqdm.auto import tqdm

from arxiv_reader.latex_utils import latex2speech

DATE_FMT = "%d-%m-%Y"
ARXIV_QUERY = "(%(categories)s) AND lastUpdatedDate:[%(start)s TO %(end)s]"
ASTRO_CATEGORIES = ("cat:astro-ph.GA", "cat:astro-ph.CO", "cat:astro-ph.EP", "cat:astro-ph.HE", "cat:astro-ph.IM", "cat:astro-ph.SR")

logger = logging.getLogger(__name__)
stream = sys.stderr
sh = logging.StreamHandler(stream=stream)
logger.addHandler(sh)
logger.setLevel(logging.INFO)
local_tz = tz.tzlocal()


class Config(NamedTuple):
    author: str
    author_email: str
    copyright: str
    base_url: str


candidates = [
    Path(".") / "config.toml",
    Path("~").expanduser() / ".config" / "arxiv_reader" / "config.toml",
]
for candidate in candidates:
    if candidate.exists():
        config = Config(**toml.load(candidate)["arxiv_reader"])
        break
else:
    config = Config(
        author="Noone",
        author_email="noone@example.com",
        copyright="None",
        base_url="example.com",
    )


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


def get_start_end(base_date: Optional[str]) -> Tuple[datetime, datetime, datetime]:
    # ArXiV papers published on day X have been submitted between 20:00 Eastern US on
    # day X-2 and 19:59 on day X-1
    eastern_US_tz = tz.gettz("US/Eastern")

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


def pull(*, base_date: Optional[str], output: str) -> int:
    date, start_date, end_date = get_start_end(base_date)
    output_folder = get_create_output_folder(output, date)

    # No papers on Sat/Sun
    if date.weekday() in (5, 6):
        return 0

    start = start_date.strftime("%Y%m%d1400")
    end = end_date.strftime("%Y%m%d1359")

    logger.info(
        f"Querying ADS from {start_date:%d %m %Y 14:00 EDT} "
        f"to {end_date:%d %m %Y 13:59 EDT}"
    )
    q = ARXIV_QUERY % dict(categories=" OR ".join(ASTRO_CATEGORIES), start=start, end=end)
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
        normalized_title = quote(re.subn(r"_?[ /\\${}\(\)\[\]]+_?", "_", title)[0])
        logger.debug("Processing %s", normalized_title)
        if len(entry.authors) > 1:
            author_str = entry.authors[0].name + " et al. "
        else:
            author_str = entry.authors[0].name + ". "

        abstract = latex2speech(entry.summary.replace("\n", " "))
        feed_as_txt = dedent(
            f"""
        <p>“{title}”, <break time="200ms"/> by {author_str}</p>

        <p>
        {abstract}.
        </p>
        <p><emphasis level=\"strong\">This</emphasis> was “{title}”,
        <break time="200ms"/> by {author_str}.</p>
        """
        )
        path = output_folder / f"{i+1}.{normalized_title}.mp3"
        if not path.exists():
            path = tts(feed_as_txt, path)

        out_code |= set_metadata(
            path,
            PaperMetadata(
                title=title,
                authors=author_str,
                pubdate=entry.published,
                url=f"http://arxiv.org/abs/{entry.entry_id}",
                abstract=entry.summary,
            ),
        )

    return out_code


def create_rss_feed(*, output: str, max_time: int, rss_file: str, categories: List[str]) -> int:
    output_folder = Path(output)

    fg = FeedGenerator()
    fg.load_extension("podcast")
    fg.podcast.itunes_category({"cat": "Science & Medicine", "sub": "Natural Science"})
    fg.podcast.itunes_author(config.author)
    fg.podcast.itunes_explicit("no")
    fg.podcast.itunes_complete("no")

    fg.title("Daily Arxiv Papers")
    fg.description("Daily astrophysics papers on the arxiv.")
    fg.link(href=f"{config.base_url}/podcast.rss", rel="alternate")
    fg.language("en")
    year = datetime.now().year
    fg.copyright(f"Copyright (c) {year} {config.copyright}")
    fg.author({"name": config.author, "email": config.author_email})
    eastern_US_tz = tz.gettz("US/Eastern")

    max_time_dt = datetime.now(tz=local_tz) - timedelta(max_time)

    for out in sorted(output_folder.glob("*")):
        if out.is_file():
            continue
        date_str = out.name
        dt = datetime.strptime(date_str, DATE_FMT).astimezone(eastern_US_tz)
        logger.info("Creating feed for date %s", date_str)
        for file in sorted(out.glob("*.mp3")):
            title = unquote(" ".join(file.name.replace("_", " ").split(".")[1:-1]))
            metadata = get_metadata(file)
            if dt < max_time_dt:
                continue
            url = f"{config.base_url}/{date_str}/{quote(file.name)}"
            logger.info("Found mp3 file %s", file)

            fe = fg.add_entry()
            fe.id(url)
            fe.pubDate(dt)
            fe.title(title)
            fe.guid(url, permalink=True)
            if metadata.authors:
                content = [f"{title} by {metadata.authors} on {dt}"]
            else:
                content = [f"{title} on {dt}"]

            if metadata.abstract:
                content.append(metadata.abstract.replace("\n", "\n"))
            if metadata.url:
                content.append(f"arXiv: {metadata.url}")
            fe.description("\n".join(content))
            fe.enclosure(url, length=str(file.stat().st_size), type="audio/mpeg")

    podcast_file = output_folder / rss_file
    logger.info("Writing podcast in %s", podcast_file)
    fg.rss_str(pretty=True)
    fg.rss_file(str(podcast_file))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-o", "--output", help="Output folder", type=str, default="output"
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers()
    parser_pull = subparsers.add_parser("pull")
    parser_pull.add_argument(
        "-d", "--date", help="Date to query. Defaults to today.", default=None, type=str
    )
    parser_pull.set_defaults(func=pull)

    parser_rss = subparsers.add_parser("rss")
    parser_rss.add_argument(
        "-f", "--rss-file", help="Output RSS file", type=str, default="podcast.xml"
    )
    parser_rss.add_argument(
        "--categories", default=ASTRO_CATEGORIES, nargs="+", type=str,
        help=(
            "Categories to include in the RSS feed. Defaults to all astronomy "
            "categories. See https://arxiv.org/ for the possible ones."
        )
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

    return args.func(**vars(args))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
