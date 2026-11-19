# Breast Cancer Cell Classification

This was my first encounter with the wonder of neural networks. Long before the generative AI boom, I wanted to understand how a network could learn, so I built one in F# and experimented with breast cancer cell classification.

The code is rough, and I would write it very differently today. I've kept it as it was for sentimental reasons: it reminds me of the curiosity that started my journey into neural networks.

This is a historical learning project, not a medical tool. It is not intended for diagnosis or clinical use.

## What is here

- `script.fsx` contains a feed-forward neural network and backpropagation implementation, along with data loading and training code.
- `dataset/` contains the data used by the experiment.
- `dat/` contains experiment outputs.
- `web/` contains the accompanying web application.

The network uses two hidden layers and was written in F# using list operations and record types. The script also includes an accuracy calculation; its results should not be interpreted as independently validated clinical performance.

## Running the historical experiment

The script was written for an older F# environment. Before running it in F# Interactive, inspect the paths in `script.fsx` and replace the hard-coded Windows dataset and output paths with locations on your machine. The original setup used Visual Studio 2015; compatibility with current tooling has not been verified.

## Data and credits

The project uses a breast cancer dataset attributed to Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian, via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). See `dataset/` for the files included in this repository.

## License

[MIT](LICENSE)
