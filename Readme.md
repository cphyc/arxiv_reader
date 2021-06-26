# Arxiv Reader

Some people like to kickstart their day as a researcher by reading the papers appearing on the Arxiv.
Some wish they could _listen_ to these abstracts while commuting or on their way to work.
This projects aims at doing exactly this.

## How does it work?

This script is actually simple and simply brings together different already-existing tools.
It fetches the paper (title, authors, abstract) from the ArXiV using the public API.
A text-only description of the paper is then generated (title, author, abstract).
The final step is to feed the text-only description to a text-to-speech synthesis (TTS) app.
For now, the only supported TTS is [Google's](https://cloud.google.com/text-to-speech) but it would be simple to use another one instead.

## Installing

The script requires a working installation of Python 3 and or the Google cloud sdk.
To install the script, download it (using e.g. `git clone`) and in the downloaded folder, install the package with
```bash
pip install .
```
That's it!
Note that before using the package, you need to have set up a [Google TTS account](https://cloud.google.com/text-to-speech/docs/quickstart-protocol).
*Before* running the script, do not forget to export your Google application credentials.
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
```

To run the script, now simply do
```bash
arxiv-reader -d "two days ago"
arxiv-reader -d "01 June 2021"
```
