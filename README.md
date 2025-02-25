# Roman Urdu Poetry Generator

An AI-powered poetry generation application that creates Roman Urdu poetry using deep learning. Built with Streamlit and PyTorch, this application generates creative and unique Urdu poetry in Roman script.

![App Screenshot](logo.png)

## Features

- Real-time poetry generation using LSTM neural networks
- Interactive web interface built with Streamlit
- Customizable generation parameters:
  - Adjustable text length
  - Temperature control for creativity
  - Custom seed text input


## Usage

1. Run the Streamlit application:
```bash
streamlit run streamlit_RomanPoetryGen.py
```

2. Open your web browser and navigate to the provided local URL
3. Enter a seed text in Roman Urdu
4. Adjust the generation parameters:
   - Text length (50-1000 characters)
   - Temperature (0.5-1.5)
5. Click "Generate Poetry" to create new poetry

## Technical Details

- Model Architecture:
  - Embedding Layer (128 dimensions)
  - 3-layer LSTM (256 hidden units)
- Device Support: CPU and CUDA-enabled GPUs

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- pandas
- unidecode

## File Structure

```
├── streamlit_RomanPoetryGen.py
├── poetGenModel.pth
├── Roman-Urdu-Poetry.csv
├── logo.png
└── requirements.txt
```