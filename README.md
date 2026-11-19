# Breast Cancer Cell Classification

**A machine-learning experiment in breast cancer cell classification, written in F#.**

This was my first encounter with the wonder of neural networks. Long before the generative AI boom, I wanted to understand how a network could learn, so I built one and experimented with classifying breast cancer cells.

The code is rough, and I would write it very differently today. I've kept it as it was for sentimental reasons: it reminds me of the curiosity that started my journey into neural networks.

This is a historical learning project. It is **not a medical device** and is not intended for diagnosis or clinical use.

## About the project

The project includes a feed-forward neural network written in F#, with two hidden layers and a backpropagation training implementation. It also includes a dataset, experiment outputs, and an accompanying web application.

I wrote the network and its training steps to explore the underlying ideas, rather than relying on a machine-learning framework.

## Repository contents

- [`script.fsx`](script.fsx) — neural network, data loading, training, and an accuracy calculation.
- [`dataset/`](dataset/) — dataset files used by the experiment.
- [`dat/`](dat/) — experiment outputs.
- [`web/`](web/) — the accompanying web application.

## Installation

### For users

If you want to explore the application, see the [web application directory](web/).

### For developers and researchers

The original development setup used [Visual Studio 2015](https://visualstudio.microsoft.com/downloads/) with F# support. You can also inspect and run the script using a suitable F# Interactive environment.

This is an older project, and compatibility with current tooling has not been verified.

## Usage

### Running the training script

The model can be retrained by running [`script.fsx`](script.fsx) in F# Interactive.

Before running it, update the hard-coded Windows paths in the script to point to the dataset and output directories on your own machine.

The script includes an accuracy calculation, but the results should not be interpreted as independently validated clinical performance.

### Exploring the web application

See [`web/`](web/) for the application files and [`dataset/`](dataset/) for the data included with this repository.

## Contributions

This repository is preserved as an early learning project, but you are welcome to explore it.

- [Issue tracker](https://github.com/ronnelreposo/breast-cancer-cell/issues)
- [Source code: F# script](script.fsx)
- [Dataset](dataset/)

## Data and credits

The project uses a breast cancer dataset attributed to:

1. **Dr. William H. Wolberg**, General Surgery Department, University of Wisconsin Clinical Sciences Center.
2. **W. Nick Street**, Computer Sciences Department, University of Wisconsin.
3. **Olvi L. Mangasarian**, Computer Sciences Department, University of Wisconsin.

The dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).

## Support

If you have questions about the project, you can open an issue or contact [@ronnelreposo](https://github.com/ronnelreposo).

## License

This project is licensed under the [MIT License](LICENSE).
