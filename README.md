# spatialStrings - 15-112 Final Project

- Place looping virtual orchestral strings instruments in three-dimensions using intuitive hand gestures
- Hear the instruments with spatial audio that updates in real-time
- See sprites that visually represent the spatial position of the instruments

**Video Demo:** WIP

## Installation:

1. Clone the repository

```bash
git clone https://github.com/jo-ljacob/spatialStrings
cd spatialStrings
```

2. Install Git LFS (required for large HRTF file)

```bash
git lfs install
git lfs pull
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the project

```bash
python main.py
```

## Controls:

- Choose an instrument by clicking an icon on the left sidebar
- Place an instrument by pinching all fingers together
- Clear all instruments or pop last placed instrument using the respective buttons
- Toggle if an instrument is placed when fingers touch
- Adjust the volume using the slider
- Simulate 5 random instruments in random locations using the Random button

**Note:** For the best experience, use headphones to hear the spatial audio correctly.

## Acknowledgements

I would like to thank my 15-112 instructor and teaching assistants for their guidance and support throughout this project.

Parts of the code that were outside of my scope were written with the assistance of AI tools such as [ChatGPT](https://chatgpt.com/) and [Claude](https://claude.ai/). All final implementations and testing were completed by me.

GUI was heavily inspired by [xLogicCircuits](https://math.hws.edu/eck/js/xLogicCircuits/xLogicCircuits.html).
