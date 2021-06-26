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

The first step for the package to work is to download it (using e.g. `git clone`). In the folder, then install the requirements
```bash
pip install -r Requirements.txt
```
You can then 

# TODO list

